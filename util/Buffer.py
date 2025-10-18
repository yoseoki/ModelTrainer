import os
os.path.abspath(os.path.dirname(__file__))
import numpy as np
import cupy as cp
from PCA import WeightPCA
from SUBSPACE import SubspaceDiff

class OrthBasisBuffer:
    
    def __init__(self, model, save_layers, non_save_layers, square_flag, salt_policy, mag_mode="orth", reshape_mode="in_dim"):
        self.model = model
        self.save_layers = save_layers
        self.non_save_layers = non_save_layers
        self.buffer = {}
        self.baseMagnitude = {}
        self.square_flag = square_flag
        self.salt_policy = salt_policy
        self.mag_mode = mag_mode
        self.reshape_mode = reshape_mode

        self.pcaTool = WeightPCA()
        self.smTool = SubspaceDiff()

        for name, _ in self.model.named_parameters(): # iteration : each parameter
            
            saveFoundFlag = False
            nonSaveFoundFlag = False
            for element in self.save_layers:
                if element in name: saveFoundFlag = True
            for element in self.non_save_layers:
                if element in name: nonSaveFoundFlag = True
            if not saveFoundFlag or nonSaveFoundFlag: continue

            print("found! : {}".format(name))

            self.buffer[name] = []
            self.baseMagnitude[name] = 1
            
    def getAplhas(self, model_name):

        if self.salt_policy == "none":
            if model_name.upper() == "RESNET18_SMALL": return [1, 1, 1, 1, 1, 1]
            elif model_name.upper() == "VGG16": return [1, 1, 1, 1, 1, 1]
            elif model_name.upper() == "VIT": return [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            elif model_name.upper() == "LENET": return [1, 1, 1, 1, 1]
        elif self.salt_policy == "direct":
            if model_name.upper() == "RESNET18_SMALL": return [16, 16, 8, 4, 2, 20]
            elif model_name.upper() == "VGG16": return [4, 2, 1, 0.5, 0.5, 10]
            elif model_name.upper() == "VIT": return [6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 17.5]
        elif self.salt_policy == "channel":
            if model_name.upper() == "RESNET18_SMALL": return [45, 16, 8, 4, 2, 3.5]
            elif model_name.upper() == "VGG16": return [4, 2, 1, 0.5, 0.5, 1]
            elif model_name.upper() == "VIT": return [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.5]
        elif self.salt_policy == "xavier":
            if model_name.upper() == "RESNET18_SMALL": return [4, 4, 3, 2, 1.5, 4]
            elif model_name.upper() == "VGG16": return [2, 1.5, 1, 0.7, 0.7, 2]
            elif model_name.upper() == "VIT": return [3, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 3.5]
        else:
            return None

    def update(self):
        """
        지정된 step / epoch 마다 "대표 레이어" 의 웨이트로
        subspace(orthonormal basis) 를 만들어
        buffer 에 append 하는 메소드.
        즉, 말 그대로 buffer 를 update 하는 메소드이다.
        """
        print()

        for name, param in self.model.named_parameters(): # iteration : each parameter
            
            # "파라미터를 저장하는 대상 레이어" 중에서
            # "대표 레이어" 인지를 확인
            # 그렇지 않으면 continue
            saveFoundFlag = False
            nonSaveFoundFlag = False
            for element in self.save_layers:
                if element in name: saveFoundFlag = True
            for element in self.non_save_layers:
                if element in name: nonSaveFoundFlag = True
            if not saveFoundFlag or nonSaveFoundFlag: continue

            param_copied = param.detach().clone()
            param_cupy = cp.asarray(param_copied.cpu().numpy())

            # Current Settings
            # Data centerizing : False
            if "feature" in name or "conv" in name or "cnn" in name: # convolution layer
                out_channels = param_cupy.shape[0]
                in_channels = param_cupy.shape[1]
                ker_size = param_cupy.shape[2] * param_cupy.shape[3]

                if self.reshape_mode == "in_dim" :
                    tmp = cp.swapaxes(param_cupy, 0, 1)
                    dataArray = cp.transpose(cp.reshape(tmp, (in_channels, out_channels * ker_size)))
                    _, basis = self.pcaTool.pca_basic(dataArray)
                elif self.reshape_mode == "out_dim" :
                    dataArray = cp.transpose(cp.reshape(param_cupy, (out_channels, in_channels * ker_size)))
                    _, basis = self.pcaTool.pca_basic(dataArray)
                elif self.reshape_mode == "ker_dim" : 
                    dataArray = cp.transpose(cp.reshape(param_cupy, (in_channels * out_channels, ker_size)))
                    _, basis = self.pcaTool.pca_basic(dataArray)
                elif self.reshape_mode == "flatten" :
                    dataArray = cp.ravel(param_cupy)
                    basis = dataArray / cp.linalg.norm(dataArray)

                # print distance
                # --- This code is for checking the distance between out_dim way and in_dim way ---
                # if out_channels == in_channels:
                #     tmp = cp.swapaxes(param_cupy, 0, 1)
                #     dataArray = cp.transpose(cp.reshape(tmp, (in_channels, out_channels * ker_size)))
                #     _, basis1 = self.pcaTool.pca_basic(dataArray)
                #     dataArray = cp.transpose(cp.reshape(param_cupy, (out_channels, in_channels * ker_size)))
                #     _, basis2 = self.pcaTool.pca_basic(dataArray)
                #     print("[{}] distance of out-channel and in channel way : ".format(name), end="")
                #     print(self.smTool.calc_magnitude(basis1, basis2))
            else: # fc layer
                out_dims = param_cupy.shape[0]
                in_dims = param_cupy.shape[1]

                # for fc layer, It is impossible to select axis for deciding number of data
                if self.reshape_mode == "flatten" :
                    dataArray = cp.ravel(param_cupy)
                    basis = dataArray / cp.linalg.norm(dataArray)
                else:
                    if in_dims < out_dims:
                        dataArray = param_cupy
                        _, basis = self.pcaTool.pca_basic(dataArray)
                    else:
                        dataArray = cp.transpose(param_cupy)
                        _, basis = self.pcaTool.pca_basic(dataArray)

            # serialize
            # --- This code is for checking the performance of vectorized orthnormal basis ---
            # basis = cp.reshape(basis, (-1, 1), order='F')
            # basis = basis / cp.linalg.norm(basis)

            # conv / fc 각각 다른 방법으로 orthonormal basis 를 만들어서 buffer 에 append
            self.buffer[name].append(basis)

    def calc_magnitude(self):
        """
        가장 최근에 append 된 3개의 subspace 를 사용해서
        "대표 레이어" 마다 "magnitude 정보를 활용한
        acceleration coef" 를 계산,
        순서 리스트 형태로 반환하는 메소드.
        이 정보는 learning rate 를 accelerate 하는 데 사용된다.
        """

        if len(self.buffer[next(iter(self.buffer))]) < 3:
            print("Not enough basis in buffer!")
            return None
        
        result = []
        
        for name, _ in self.model.named_parameters(): # iteration : each parameter
            
            # "파라미터를 저장하는 대상 레이어" 중에서
            # "대표 레이어" 인지를 확인
            # 그렇지 않으면 continue
            saveFoundFlag = False
            nonSaveFoundFlag = False
            for element in self.save_layers:
                if element in name: saveFoundFlag = True
            for element in self.non_save_layers:
                if element in name: nonSaveFoundFlag = True
            if not saveFoundFlag or nonSaveFoundFlag: continue

            # 계산 대상 : 바로 직전에 계산된 3개의 orthonormal basis
            basis_prev = self.buffer[name][-3]
            basis_cur = self.buffer[name][-2]
            basis_next = self.buffer[name][-1]

            if "orth" in self.mag_mode:  # (2nd magnitude 의) along, orth component 를 계산
                along, orth = self.smTool.calc_2nd_magnitude_decomposed(basis_prev, basis_cur, basis_next)
                along = along.get()
                orth = orth.get()
                # # for debugging
                # !
                # print("(({:.4f}, {:.4f}))".format(along, orth), end="  ")
            else:   
                if "1" == self.mag_mode[-1]: # 1st magnitude 를 계산
                    mag = self.smTool.calc_magnitude(basis_prev, basis_next)
                    mag = mag.get()
                else: # 2nd magnitude 를 계산
                    _, k = self.smTool.calc_karcher_subspace(basis_prev, basis_next)
                    mag = self.smTool.calc_magnitude(basis_cur, k)
                    mag = mag.get()
                # for debugging
                # !
                # print("({:.4f})".format(mag), end="  ")      
            
            # 예전 계산 방식 1
            # orth component 의 logit 을 사용
            # 단, 한 쪽의 점유율이 90% 이상이 되면 0.9 와 0.1 을 사용
            # logit 의 explosion 을 방지
            # if along + orth < 1e-4:
            #     logit = orth
            # else:
            #     orth_tmp = min(orth / (along + orth), 0.9)
            #     along_tmp = max(along / (along + orth), 0.1)
            #     logit = orth_tmp / along_tmp

            #  예전 계산 방식 2
            # orth component 를 그대로 사용
            # result.append(orth)

            # 현재 계산 방식
            # 초기 magnitude 와 비교해서 얼마나 커졌는지 비를 사용
            # 단, 폭발하기 쉬우므로 제곱근 값을 사용(option)
            # 또한, "along component 가 충분히 크면서" orth component 가 작은 경우
            # 아직 수렴하지 않았다고 판단, 조기 수렴 방지를 위해
            # orth component = 1e-4 를 사용
            if self.square_flag:
                if "orth" in self.mag_mode:
                    if abs(orth) < 1e-4 and abs(along) > 1e-4: logit = np.sqrt(1e-4 / self.baseMagnitude[name])
                    else: logit = np.sqrt(abs(orth) / self.baseMagnitude[name])
                else:
                    logit = np.sqrt(abs(mag) / self.baseMagnitude[name])
            else:
                if "orth" in self.mag_mode:
                    if abs(orth) < 1e-4 and abs(along) > 1e-4: logit = 1e-4 / self.baseMagnitude[name]
                    else: logit = abs(orth) / self.baseMagnitude[name]
                else:
                    logit = abs(mag) / self.baseMagnitude[name]

            result.append(logit)
            self.buffer[name].pop(0)
        
        # for debugging
        # !
        # print()

        return result
    
    def set_basis(self):
        """
        초기 3개의 subspace 를 사용해서
        "대표 레이어" 마다 "acceleration coef 의 기준점" 을 계산,
        클래스 내 사전형 변수에 저장하는 메소드.
        이 정보는 accelerate coef 의 기준점으로써,
        적절히 scaling 하는 데에 사용된다.
        """

        if len(self.buffer[next(iter(self.buffer))]) < 3:
            print("Not enough basis in buffer!")
            return None
        
        for name, _ in self.model.named_parameters(): # iteration : each parameter
            
            # "파라미터를 저장하는 대상 레이어" 중에서
            # "대표 레이어" 인지를 확인
            # 그렇지 않으면 continue
            saveFoundFlag = False
            nonSaveFoundFlag = False
            for element in self.save_layers:
                if element in name: saveFoundFlag = True
            for element in self.non_save_layers:
                if element in name: nonSaveFoundFlag = True
            if not saveFoundFlag or nonSaveFoundFlag: continue

            # 계산 대상 : 바로 직전에 계산된 3개의 orthonormal basis
            # 반드시 calc_2nd_magnitude 전에 호출되어 기준점을 미리 계산해야 함.
            basis_prev = self.buffer[name][-3]
            basis_cur = self.buffer[name][-2]
            basis_next = self.buffer[name][-1]

            if "orth" in self.mag_mode:  # (2nd magnitude 의) along, orth component 를 계산
                along, orth = self.smTool.calc_2nd_magnitude_decomposed(basis_prev, basis_cur, basis_next)
                along = along.get()
                orth = orth.get()
                # # for debugging
                # !
                # print("(({:.4f}, {:.4f}))".format(along, orth), end="  ")
            else:   
                if "1" == self.mag_mode[-1]: # 1st magnitude 를 계산
                    mag = self.smTool.calc_magnitude(basis_prev, basis_next)
                    mag = mag.get()
                else: # 2nd magnitude 를 계산
                    _, k = self.smTool.calc_karcher_subspace(basis_prev, basis_next)
                    mag = self.smTool.calc_magnitude(basis_cur, k)
                    mag = mag.get()
                # for debugging
                # !
                # print("({:.4f})".format(mag), end="  ")

            # 기준점 : 초기 orth(mag) component of 2nd magnitude
            # 단, 1e-4 보다 작은 값은 오차라고 보고 사용하지 않음
            if "orth" in self.mag_mode: logit = orth
            else: logit = mag

            if logit < 1e-4: logit = 1e-4
            self.baseMagnitude[name] = logit

    def set_basis_manually(self, basisList):
        """
        직접 "기준점" 을 계산하지 않고 인자로써 "기준점 배열" 을 전달받아
        클래스 내 사전형 변수에 저장하는 메소드.
        이 정보는 accelerate coef 의 기준점으로써,
        적절히 scaling 하는 데에 사용된다.
        """

        cnt = 0

        for name, _ in self.model.named_parameters(): # iteration : each parameter
            
            # "파라미터를 저장하는 대상 레이어" 중에서
            # "대표 레이어" 인지를 확인
            # 그렇지 않으면 continue
            saveFoundFlag = False
            nonSaveFoundFlag = False
            for element in self.save_layers:
                if element in name: saveFoundFlag = True
            for element in self.non_save_layers:
                if element in name: nonSaveFoundFlag = True
            if not saveFoundFlag or nonSaveFoundFlag: continue

            self.baseMagnitude[name] = basisList[cnt]
            cnt += 1

    def clear_buffer(self):

        for name, _ in self.model.named_parameters(): # iteration : each parameter
            
            # "파라미터를 저장하는 대상 레이어" 중에서
            # "대표 레이어" 인지를 확인
            # 그렇지 않으면 continue
            saveFoundFlag = False
            nonSaveFoundFlag = False
            for element in self.save_layers:
                if element in name: saveFoundFlag = True
            for element in self.non_save_layers:
                if element in name: nonSaveFoundFlag = True
            if not saveFoundFlag or nonSaveFoundFlag: continue

            self.buffer[name].clear()
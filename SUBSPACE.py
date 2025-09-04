import cupy as cp

class SubspaceDiff():
    def __init__(self):
        pass

    def calc_sum_subspace(self, basis1, basis2):

        G = basis1@cp.transpose(basis1) + basis2@cp.transpose(basis2)
        a, l = cp.linalg.eigh(G)
        alphas = a[::-1]
        lambdas = l[:, ::-1]
        alphas, lambdas = self.adjust_eig(alphas, lambdas)

        index = 0
        for j, element in enumerate(alphas):
            if element < 0.0:
                index = j
                break

        return [alphas[0:index], lambdas[:,0:index]]

    def calc_overlap_subspace(self, basis1, basis2, isVerbose=False):

        G = basis1@cp.transpose(basis1) + basis2@cp.transpose(basis2)
        a, l = cp.linalg.eigh(G)
        alphas = a[::-1]
        lambdas = l[:, ::-1]
        # alphas, lambdas = self.adjust_eig(alphas, lambdas)

        index = 0
        for i, a in enumerate(alphas):
            if a < 2.0 - 1e-4:
                index = i
                break

        if isVerbose: print("{}-dimension is overlapped!".format(index))

        return [alphas[0:index], lambdas[:,0:index]]
    
    def calc_karcher_subspace(self, basis1, basis2):

        G = basis1@cp.transpose(basis1) + basis2@cp.transpose(basis2)
        a, l = cp.linalg.eigh(G)
        alphas = a[::-1]
        lambdas = l[:, ::-1]
        alphas, lambdas = self.adjust_eig(alphas, lambdas)

        index = 0
        for i, a in enumerate(alphas):
            if a < 1.0:
                index = i
                break

        return [alphas[0:index], lambdas[:,0:index]]
    
    def calc_diff_subspace(self, basis1, basis2):

        G = basis1@cp.transpose(basis1) + basis2@cp.transpose(basis2)
        a, l = cp.linalg.eigh(G)
        alphas = a[::-1]
        lambdas = l[:, ::-1]
        alphas, lambdas = self.adjust_eig(alphas, lambdas)
        # print(alphas)

        index_start = 0
        for i, a in enumerate(alphas):
            if a < 1.0:
                index_start = i
                break

        index_end = 0
        for i, a in enumerate(alphas):
            if a < 1e-5:
                index_end = i
                break

        print("dim : {}".format(index_end - index_start))
        # print(index_start)
        # print(index_end)

        return [alphas[index_start:index_end], lambdas[:,index_start:index_end]]
    
    def adjust_eig(self, eigenvalues, eigenvectors, epsilon=1e-9):
        eig_num = eigenvalues.shape[0]

        eigenvalues_new = []
        eigenvectors_new = []

        for i in range(eig_num):
            value = eigenvalues[i]
            if value >= 2.0 or (value < 2.0 and value > 2.0 - epsilon): # dims overlapped
                eigenvalues_new.append(2.0)
                eigenvectors_new.append(eigenvectors[:,i])
            elif (value < epsilon and value > 0.0) or value <= 0.0: # dims cannot express
                eigenvalues_new.append(0.0)
                eigenvectors_new.append(eigenvectors[:,i])
            else:
                eigenvalues_new.append(value.get())
                eigenvectors_new.append(eigenvectors[:,i])

        return [cp.array(eigenvalues_new), cp.transpose(cp.array(eigenvectors_new))]

    def calc_magnitude(self, basis1, basis2, isVerbose=False, overlapFlag=False):
        
        if not overlapFlag:

            _, basis_overlap = self.calc_overlap_subspace(basis1, basis2, isVerbose=isVerbose)

            P1 = basis1@cp.transpose(basis1) - basis_overlap@cp.transpose(basis_overlap)
            P2 = basis2@cp.transpose(basis2) - basis_overlap@cp.transpose(basis_overlap)

            alpha1, basis1_revised = cp.linalg.eigh(P1)
            alpha1 = alpha1[::-1]
            basis1_revised = basis1_revised[:,::-1]
            index_1 = 0
            for i, a in enumerate(alpha1):
                if abs(a) < 1e-1 :
                    index_1 = i
                    break
            basis1_revised = basis1_revised[:,:index_1]

            alpha2, basis2_revised = cp.linalg.eigh(P2)
            alpha2 = alpha2[::-1]
            basis2_revised = basis2_revised[:,::-1]
            index_2 = 0
            for j, a in enumerate(alpha2):
                if abs(a) < 1e-1 :
                    index_2 = j
                    break
            basis2_revised = basis2_revised[:,:index_2]

        else:
            basis1_revised = basis1
            basis2_revised = basis2
        # print("index1 : {}".format(index_1))
        # print(alpha1)
        # print("index2 : {}".format(index_2))
        # print(alpha2)

        G = cp.transpose(basis1_revised)@basis2_revised
        _, s, _ = cp.linalg.svd(G)
        if isVerbose:
            for ele_s in s:
                print("{:.2f}, ".format(ele_s), end="")
            print()

        return 2 * (len(s) - cp.sum(s))
    
    def calc_similarity(self, basis1, basis2, isVerbose=False):
        
        _, basis_overlap = self.calc_overlap_subspace(basis1, basis2, isVerbose=isVerbose)
        tmp = basis_overlap.shape[1]

        P1 = basis1@cp.transpose(basis1) - basis_overlap@cp.transpose(basis_overlap)
        P2 = basis2@cp.transpose(basis2) - basis_overlap@cp.transpose(basis_overlap)

        alpha1, basis1_revised = cp.linalg.eigh(P1)
        alpha1 = alpha1[::-1]
        basis1_revised = basis1_revised[:,::-1]
        index_1 = 0
        for i, a in enumerate(alpha1):
            if abs(a) < 1- 1e-4 :
                index_1 = i
                break
        basis1_revised = basis1_revised[:,:index_1]

        alpha2, basis2_revised = cp.linalg.eigh(P2)
        alpha2 = alpha2[::-1]
        basis2_revised = basis2_revised[:,::-1]
        index_2 = 0
        for j, a in enumerate(alpha2):
            if abs(a) < 1- 1e-4 :
                index_2 = j
                break
        basis2_revised = basis2_revised[:,:index_2]

        G = cp.transpose(basis1_revised)@basis2_revised
        _, s, _ = cp.linalg.svd(G)
        if isVerbose: print(s)

        return cp.sum(s) + tmp
    
    def calc_1st_magnitude_decomposed(self, basis1, basis2, basis3):

        W_tmp = cp.concatenate((basis1, basis3), axis=1)
        l, W = cp.linalg.eigh(W_tmp@cp.transpose(W_tmp))
        l = l[::-1]
        W = W[:,::-1]
        idx = 0
        for i, ele_l in enumerate(l):
            if ele_l < 1e-4:
                idx = i
                break
        W = W[:,:idx]
        _, s, _ = cp.linalg.svd(cp.transpose(W)@basis2)
        U, _ = cp.linalg.qr(cp.transpose(W)@basis2)
        basis2_prime = W@U

        mag_orth = 2 * (len(s) - cp.sum(s))
        mag_along = self.calc_magnitude(basis2_prime, basis1)

        return [mag_along, mag_orth]

    def calc_2nd_magnitude_decomposed(self, basis1, basis2, basis3):

        W_tmp = cp.concatenate((basis1, basis3), axis=1)
        W, _ = cp.linalg.qr(W_tmp)
        _, M = self.calc_karcher_subspace(basis1, basis3)
        _, s, _ = cp.linalg.svd(cp.transpose(W)@basis2)
        U, _ = cp.linalg.qr(cp.transpose(W)@basis2)
        basis2_prime = W@U

        mag_orth = 2 * (len(s) - cp.sum(s))
        mag_along = self.calc_magnitude(basis2_prime, M)

        return [mag_along, mag_orth]
    
    def calc_rbf_magnitude(self, alphas1, alphas2, km):
        _, s, _ = cp.linalg.svd(cp.transpose(alphas2) @ cp.transpose(km) @ alphas1)
        return 2 * (len(s) - cp.sum(s))
    
class Grassmannian():
    def __init__(self):
        pass
    
    def get_grassmannian(self, basis):
        P = basis@cp.transpose(basis)
        return P
    
    def get_symmetric(self, X):
        symX = 0.5 * (X + cp.transpose(X))
        return symX
    
    def get_tangent_projection(self, X, D):
        I = cp.eye(X.shape[0])
        tmp = X @ self.get_symmetric(D) @ (I - X)
        return 2 * self.get_symmetric(tmp)
    
    def calc_grassmannian_inner_product(self, zeta, eta):
        return cp.trace(cp.transpose(zeta)@eta)
        
    def calc_grassmannian_norm(self, zeta):
        return cp.sqrt(cp.trace(cp.transpose(zeta)@zeta))
    
    def normalize(self, X):
        return X / self.calc_grassmannian_norm(X)
    
    def logarithmic_mapping(self, K, X):

        YTU = cp.transpose(X)@K

        _, Q_tilde = cp.linalg.eigh(YTU@cp.transpose(YTU))
        Q_tilde = Q_tilde[:,::-1]
        _, R_tilde = cp.linalg.eigh(cp.transpose(YTU)@YTU)
        R_tilde = R_tilde[:,::-1]
        Y_prime = X @ Q_tilde @ cp.transpose(R_tilde)

        result = (cp.eye(K.shape[0]) - K@cp.transpose(K)) @ Y_prime
        _, Q = cp.linalg.eigh(result@cp.transpose(result))
        Q = Q[:,::-1]
        _, R = cp.linalg.eigh(cp.transpose(result)@result)
        R = R[:,::-1]
        _, sigma, _ = cp.linalg.svd(result)

        return Q[:,:sigma.shape[0]] @ cp.diag(cp.arcsin(sigma)) @ cp.transpose(R)
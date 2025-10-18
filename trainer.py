import sys, os
os.path.abspath(os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import random as rd
import numpy as np
from util import CNNModel as nmc
from util import Buffer as buf
import matplotlib.pyplot as plt
import math
import json
from overrides import overrides
from safetensors.torch import save_file, safe_open
import cupy as cp
from PCA import WeightPCA
from SUBSPACE import SubspaceDiff
from itertools import chain

def smoothing(signal, w_size=3):

	result = []
	totalNum = len(signal)
	offset = w_size // 2
	for i in range(offset, totalNum - offset):
		part = signal[i-1:i+2]
		part.sort()
		result.append(part[offset])
	
	return result

# for class which inherit modelLoader class(ex. UNetLoader, Resnet18Loader, ...)
# need to implement _build_new_model method, _build_non_save_layers method
# and only need prefix_w argument, which indicates where to save models
# and can return their model by get_model method, can save their model by save_weight method
class modelLoader:

	def __init__(self, prefix_w, num_classes):
		self.model = None
		self.non_save_layers = None
		self.prefix_w = prefix_w
		self.num_classes = num_classes

	def _build_model(self):
		if not self.non_save_layers:
			self._build_non_save_layers()

		if not self.model:
			self._build_new_model()

	def _build_new_model(self):
		pass

	def _build_non_save_layers(self):
		pass

	def get_model(self):
		if not self.model:
			self._build_model()
		
		return self.model

	def save_weight(self, is_verbose=False):

		flat = nn.Flatten(start_dim=0)
		counter = 0
		if is_verbose: print()

		for name, param in self.model.named_parameters(): # iteration : each parameter
			
			nonSaveFoundFlag = False
			for element in self.non_save_layers:
				if element in name: nonSaveFoundFlag = True
			if nonSaveFoundFlag: continue

			if is_verbose:
				print("{} : {} | {}".format(counter+1, name, param.shape))

			param_flatten = flat(param)
			param_flatten = param_flatten.detach().tolist()
			counter = counter + 1

			# save each parameter elements in divided csv file
			partitionNum = math.ceil(len(param_flatten) / 1000) 
			for m in range(partitionNum): # iteration : each divied partition in one parameter
				if not os.path.isdir("./" + self.prefix_w + "/layer{}".format(counter)):
					os.mkdir("./" + self.prefix_w + "/layer{}".format(counter))
				f = open("./" + self.prefix_w + "/layer{}/part{:04d}.csv".format(counter, m), 'a')
				if m < partitionNum - 1:
					for n in range(1000):
						if n != 0: f.write(",")
						f.write(str(param_flatten[m * 1000 + n]))
					f.write("\n")
					f.close()
				elif m == partitionNum - 1:
					for n in range(len(param_flatten) - 1000 * m):
						if n != 0: f.write(",")
						f.write(str(param_flatten[m * 1000 + n]))
					f.write("\n")
					f.close()

	def save_weight(self, epoch, is_verbose=False, is_init=False):

		counter = 0
		tensors = {}
		if is_verbose: print()

		for name, param in self.model.named_parameters(): # iteration : each parameter
			
			nonSaveFoundFlag = False
			for element in self.non_save_layers:
				if element in name: nonSaveFoundFlag = True
			if nonSaveFoundFlag: continue

			if is_verbose:
				print("{} : {} | {}".format(counter+1, name, param.shape))

			param_copied = param.detach().clone()
			tensors["{:03d}".format(counter+1)] = param_copied
			counter = counter + 1
		if is_init: save_file(tensors, "./" + self.prefix_w + "/weights_epoch_init.safetensors")
		else: save_file(tensors, "./" + self.prefix_w + "/weights_epoch{:03d}.safetensors".format(epoch))

class VGG16Loader(modelLoader):

	@overrides
	def _build_new_model(self):
		self.model = nmc.VGG16(64, num_classes=self.num_classes)

	@overrides
	def _build_non_save_layers(self):
		self.non_save_layers = ["bias", ".1.w", ".1.b", ".4.w", ".4.b", ".7.w", ".7.b"]

class Resnet18SmallLoader(modelLoader):

	@overrides
	def _build_new_model(self):
		self.model = nmc.RESNET18_small(self.num_classes)

	@overrides
	def _build_non_save_layers(self):
		self.non_save_layers = ["bias", "downsample.1", "bn"]

class Resnet18Loader(modelLoader):

	@overrides
	def _build_new_model(self):
		self.model = nmc.RESNET18(self.num_classes)

	@overrides
	def _build_non_save_layers(self):
		self.non_save_layers = ["bias", "downsample.1", "bn"]

class Resnet50Loader(modelLoader):

	@overrides
	def _build_new_model(self):
		self.model = nmc.RESNET50(in_channels=3 , num_classes=self.num_classes)

	@overrides
	def _build_non_save_layers(self):
		self.non_save_layers = ["bias", "iden", "batchnorm"]

class UnetLoader(modelLoader):

	@overrides
	def _build_new_model(self):
		self.model = nmc.UNET(self.num_classes)

	@overrides
	def _build_non_save_layers(self):
		self.non_save_layers = ["bias", ".2.w", ".2.b", ".5.w", ".5.b"]	

class ViTLoader(modelLoader):

	@overrides
	def _build_new_model(self):
		self.model = nmc.ViT(   
                in_channels=3,
                patch_size=4,
                emb_size=192,
                img_size=32,
                depth=12,
                n_classes=self.num_classes)
		
	@overrides
	def _build_non_save_layers(self):
		self.non_save_layers = ["bias", "0.projection", "fn.0.weight"]	

class LeNetLoader(modelLoader):

	@overrides
	def _build_new_model(self):
		self.model = nmc.LeNet(num_classes=self.num_classes)

	@overrides
	def _build_non_save_layers(self):
		self.non_save_layers = ["bias"]

class LeNetRevisedLoader(modelLoader):

	@overrides
	def _build_new_model(self):
		self.model = nmc.LeNetRevised(num_classes=self.num_classes)

	@overrides
	def _build_non_save_layers(self):
		self.non_save_layers = ["bias"]

# for class which inherit trainer class(ex. CIFAR10Trainer, MNISTTrainer, ...)
# need to implement load_DB method, build_num_classes method
# and can train by training method
class trainer:

	def __init__(self):
		
		self.rdSeed = 0
		self.model_name = None
		self.prefix_w = None
		self.lr = 0.0
		self.epochs = 0
		self.batch_size = 0
		self.sampling_step = 0
		self.model_loader = None
		self.optimizer = None
		self.criterion = None
		self.config = None
		self.color_list = ["blue", "orange", "red", "purple", "green", 
             "olive", "brown", "grey", "cyan", "pink",
             "navy", "lime", "black", "yellow", "crimson",
             "gold", "skyblue", "indigo", "darkgreen", "ivory",
             "blue", "orange", "red", "purple", "green", 
             "olive", "brown", "grey", "cyan", "pink",
             "navy", "lime", "black", "yellow", "crimson",
             "gold", "skyblue", "indigo", "darkgreen", "ivory",
             "blue", "orange", "red", "purple", "green", 
             "olive", "brown", "grey", "cyan", "pink",
             "navy", "lime", "black", "yellow", "crimson",
             "gold", "skyblue", "indigo", "darkgreen", "ivory"]

		self.cost_container = []
		self.val_acc_container = []

		self.num_classes = 0
		self.build_num_classes()

	def set_seed(self, rdSeed):
		self.rdSeed = rdSeed
		torch.manual_seed(rdSeed)
		rd.seed(rdSeed)

	def prepare_save_folder(self, model_name, mode=None):
		self.model_name = model_name

		if mode:
			if not os.path.isdir("./" + model_name + "_" + mode):
				os.mkdir("./" + model_name + "_" + mode)

		if mode: self.prefix_w = model_name + "_" + mode + "/" + model_name.upper() + "__" + str(self.rdSeed).zfill(4)
		else: self.prefix_w = model_name.upper() + "__" + str(self.rdSeed).zfill(4)

		if not os.path.isdir("./" + self.prefix_w):
			os.mkdir("./" + self.prefix_w)

	def parse_training_args(self, config_file):
		f = open(config_file, 'r')
		self.config = json.load(f)
		self.lr = float(self.config["learning_rate"])
		self.epochs = int(self.config["epochs"])
		self.batch_size = int(self.config["batch_size"])
		self.sampling_step = list(map(int, self.config["sampling_step"]))
		self.optimizer = self.config["optimizer"] # options : "SGD", "ADAM"
		self.criterion = self.config["criterion"] # options : "CE"
		if "offset" in self.config or "interval" in self.config:
			self.mode = "ACCELERATE"
			self.mag_mode = self.config["mag_mode"] if "mag_mode" in self.config else "orth"
			self.reshape_mode = self.config["reshape_mode"] if "reshape_mode" in self.config else "in_dim"
			self.offset = int(self.config["offset"]) if "offset" in self.config else 1
			self.interval = int(self.config["interval"]) if "interval" in self.config else 1
			if "square_flag" in self.config:
				if self.config["square_flag"].lower() == 'true':
					self.square_flag = True
				elif self.config["square_flag"].lower() == 'false':
					self.square_flag = False
			else:
				self.square_flag = False
			self.salt_policy = self.config["salt_policy"] if "salt_policy" in self.config else "none" # options : "none", "direct", "inverse", "xavier", "he"
			self.calc_policy = self.config["calc_policy"] if "calc_policy" in self.config else "epoch" # options : "epoch", "step"
			self.calc_interval = int(self.config["calc_interval"]) if "calc_interval" in self.config else 5
		else:
			self.mode = "NORMAL"
	
	def load_model(self, model_name):
		if model_name.upper() == "UNET":
			self.model_loader = UnetLoader(self.prefix_w, self.num_classes)
		elif model_name.upper() == "RESNET50":
			self.model_loader = Resnet50Loader(self.prefix_w, self.num_classes)
		elif model_name.upper() == "RESNET18":
			self.model_loader = Resnet18Loader(self.prefix_w, self.num_classes)
		elif model_name.upper() == "RESNET18_SMALL":
			self.model_loader = Resnet18SmallLoader(self.prefix_w, self.num_classes)
		elif model_name.upper() == "VGG16":
			self.model_loader = VGG16Loader(self.prefix_w, self.num_classes)
		elif model_name.upper() == "VIT":
			self.model_loader = ViTLoader(self.prefix_w, self.num_classes)
		elif model_name.upper() == "LENET_REVISED":
			self.model_loader = LeNetRevisedLoader(self.prefix_w, self.num_classes)
		elif model_name.upper() == "LENET":
			self.model_loader = LeNetLoader(self.prefix_w, self.num_classes)

		return self.model_loader.get_model()

	def load_DB(self):
		pass

	def build_num_classes(self):
		pass

	def load_optimizer(self, model):
		if self.optimizer == "SGD":
			return torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
		elif self.optimizer == "ADAM":
			return torch.optim.Adam(model.parameters(), lr=self.lr)
	
	def load_layerwise_optimizer(self, model_name, model):
		if self.optimizer == "SGD":
			if model_name.upper() == "VGG16":
				name = ["feature.0.3", "feature.1.3", "feature.2.3", "feature.3.3", "feature.4.3", "connect_fc"]
				return name, torch.optim.SGD([
					{'params': model.feature[0].parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model.feature[1].parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model.feature[2].parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model.feature[3].parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model.feature[4].parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model.connect_fc.parameters(), 'lr': self.lr, 'momentum' : 0.9}
					])
			elif model_name.upper() == "RESNET18_SMALL":
				name = ["conv1", "layer11.conv2", "layer21.conv2", "layer31.conv2", "layer41.conv2", "fc"]
				return name, torch.optim.SGD([
					{'params': chain(model.conv1.parameters(), model.bn1.parameters()), 'lr': self.lr, 'momentum' : 0.9},
					{'params': chain(model.layer11.parameters(), model.layer12.parameters()), 'lr': self.lr, 'momentum' : 0.9},
					{'params': chain(model.layer21.parameters(), model.layer22.parameters()), 'lr': self.lr, 'momentum' : 0.9},
					{'params': chain(model.layer31.parameters(), model.layer32.parameters()), 'lr': self.lr, 'momentum' : 0.9},
					{'params': chain(model.layer41.parameters(), model.layer42.parameters()), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model.fc.parameters(), 'lr': self.lr, 'momentum' : 0.9}
					])
			elif model_name.upper() == "RESNET18":
				name = ["conv1", "layer11.conv2", "layer21.conv2", "layer31.conv2", "layer41.conv2", "fc"]
				return name, torch.optim.SGD([
					{'params': chain(model.conv1.parameters(), model.bn1.parameters()), 'lr': self.lr, 'momentum' : 0.9},
					{'params': chain(model.layer11.parameters(), model.layer12.parameters()), 'lr': self.lr, 'momentum' : 0.9},
					{'params': chain(model.layer21.parameters(), model.layer22.parameters()), 'lr': self.lr, 'momentum' : 0.9},
					{'params': chain(model.layer31.parameters(), model.layer32.parameters()), 'lr': self.lr, 'momentum' : 0.9},
					{'params': chain(model.layer41.parameters(), model.layer42.parameters()), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model.fc.parameters(), 'lr': self.lr, 'momentum' : 0.9}
					])
			elif model_name.upper() == "RESNET50":
				name = ["conv1", "conv2_x.1.convseq.1.conv", "conv3_x.1.convseq.1.conv", "conv4_x.1.convseq.1.conv", "conv5_x.0.convseq.1.conv", "fc"]
				return name, torch.optim.SGD([
					{'params': model.conv1.parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model.conv2_x.parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model.conv3_x.parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model.conv4_x.parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model.conv5_x.parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model.fc.parameters(), 'lr': self.lr, 'momentum' : 0.9}
					])
			elif model_name.upper() == "VIT":
				name = []
				name.append("0.positions")
				for i in range(12):
					name.append("1.{}.1.fn.1.0.weight".format(i))
				name.append("2.2.weight")
				return name, torch.optim.SGD([
					{'params': model[0].parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model[1][0].parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model[1][1].parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model[1][2].parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model[1][3].parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model[1][4].parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model[1][5].parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model[1][6].parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model[1][7].parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model[1][8].parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model[1][9].parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model[1][10].parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model[1][11].parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model[2].parameters(), 'lr': self.lr, 'momentum' : 0.9}
					])
			elif model_name.upper() == "LENET":
				name = ["cnn1", "cnn2", "fc1", "fc2", "fc3"]
				return name, torch.optim.SGD([
					{'params': model.cnn1.parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model.cnn2.parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model.fc1.parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model.fc2.parameters(), 'lr': self.lr, 'momentum' : 0.9},
					{'params': model.fc3.parameters(), 'lr': self.lr, 'momentum' : 0.9}
					])
			
	def update_optimizer(self, optimizer, magContainer):
		
		for i, param_group in enumerate(optimizer.param_groups):
			param_group['lr'] = self.lr * magContainer[i]


	def load_criterion(self):
		if self.criterion == "CE":
			return nn.CrossEntropyLoss()

	def save_training_result(self):

		f = open("./" + self.prefix_w + "/training_config.json", 'w')
		json.dump(self.config, f)
		f.close()

		x = np.arange(1, len(self.cost_container) + 1)
		cost = np.array(self.cost_container)
		acc = np.array(self.val_acc_container) * 5

		# save cost information as csv
		f = open("./" + self.prefix_w + "/cost_container.csv", 'w')
		for n in range(len(self.cost_container)):
			if n != 0: f.write(",")
			f.write(str(self.cost_container[n]))
		f.write("\n")
		f.close()

		# save validation accuracy information as csv
		f = open("./" + self.prefix_w + "/val_acc_container.csv", 'w')
		for n in range(len(self.val_acc_container)):
			if n != 0: f.write(",")
			f.write(str(self.val_acc_container[n]))
		f.write("\n")
		f.close()

		# plot cost, validation accuracy information
		plt.plot(x, cost, color=self.color_list[-3], label="cost")
		plt.plot(x, acc, color=self.color_list[-2], label="val_acc[x 5]")
		plt.legend()
		plt.grid()
		plt.title("cost, val_acc")
		plt.savefig(self.prefix_w + "/test_result.png")
		# plt.show()
		plt.clf()

	def training(self, model_name, rdSeed, config_file_path, is_verbose=False, mode=None):
		self.set_seed(rdSeed)
		self.prepare_save_folder(model_name, mode=mode)
		self.parse_training_args(config_file_path)
		trainloader, testloader = self.load_DB()

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		model = self.load_model(model_name).to(device)

		if is_verbose:
			a = 1
			for name, param in model.named_parameters():
				print("{} : {} | {}".format(a, name, param.shape))
				a += 1
		criterion = self.load_criterion()
		if self.mode == "NORMAL":
			optimizer = self.load_optimizer(model)
		elif self.mode == "ACCELERATE":
			save_name, optimizer = self.load_layerwise_optimizer(model_name, model)
			obb = buf.OrthBasisBuffer(model, save_name, self.model_loader.non_save_layers, self.square_flag, self.salt_policy, reshape_mode=self.reshape_mode, mag_mode=self.mag_mode)
			obb.update()
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

		self.model_loader.save_weight(-1, is_verbose=is_verbose, is_init=True)

		print("========== training start! ==========")

		# save magnitude change
		totalBaseContainer = []
		totalSaltedContainer = []

		for epoch in range(self.epochs):
			print("epoch : {:03d} / {:03d} - ".format(epoch + 1, self.epochs), end="")

			if self.mode == "ACCELERATE":
				if self.calc_policy == "epoch":
					if epoch == self.offset - 1:
						obb.set_basis()
					if epoch > self.offset - 1 and (epoch - self.offset) % self.interval == 0:
						magContainer = obb.calc_magnitude()
						totalBaseContainer.append(magContainer)
						finalMagContainer = []
						alphas = obb.getAplhas(model_name)
						for alpha, comp in zip(alphas, magContainer):
							finalMagContainer.append(comp * alpha)
						for element in finalMagContainer: print("{:.4f}".format(element), end=" || ")
						print()
						totalSaltedContainer.append(finalMagContainer)
						self.update_optimizer(optimizer, finalMagContainer)
				elif self.calc_policy == "step":
					if epoch == self.offset - 1 or (epoch > self.offset - 1 and (epoch - self.offset) % self.interval == 0):
						mag_buffer = []
						for _ in range(len(save_name)): mag_buffer.append([])
						obb.clear_buffer()

			model.train()
			avg_cost = 0

			for i, data in enumerate(trainloader):

				X, Y = data

				X = X.to(device)
				Y = Y.to(device)

				optimizer.zero_grad()
				prediction = model(X)
				cost = criterion(prediction, Y)

				cost.backward()
				optimizer.step()
				avg_cost += cost.item()

				if self.mode == "ACCELERATE":
					if self.calc_policy == "step":
						if epoch == self.offset - 1 and i % self.calc_interval == 0:
							obb.update()
							if i > 2 * self.calc_interval:
								magContainer = obb.calc_magnitude()
								for j, mag in enumerate(magContainer): mag_buffer[j].append(mag)

						if epoch > self.offset - 1 and (epoch - self.offset) % self.interval == 0 and i % self.calc_interval == 0:
							obb.update()
							if i > 2 * self.calc_interval:
								magContainer = obb.calc_magnitude()
								for j, mag in enumerate(magContainer): mag_buffer[j].append(mag)

				if i in self.sampling_step:
					self.model_loader.save_weight(len(self.sampling_step)*epoch + self.sampling_step.index(i), is_verbose=is_verbose)
					if self.mode == "ACCELERATE":
						if self.calc_policy == "epoch":
							obb.update()
			
			if self.mode == "ACCELERATE":
				if self.calc_policy == "step":
					if epoch == self.offset - 1:
						_finalMagContainer = []
						for comp in mag_buffer:
							comp_smoothened = smoothing(comp)
							_finalMagContainer.append(sum(comp_smoothened))
						print("base: ", end="")
						for element in _finalMagContainer: print("{:.4f}".format(element), end=" || ")
						print()
						obb.set_basis_manually(_finalMagContainer)
					if epoch > self.offset - 1 and (epoch - self.offset) % self.interval == 0:
						finalMagContainer = []
						alphas = obb.getAplhas(model_name)
						tmp = []
						for comp in mag_buffer:
							comp_smoothened = smoothing(comp)
							tmp.append(sum(comp_smoothened))
						totalBaseContainer.append(tmp)
						for alpha, comp in zip(alphas, tmp):
							finalMagContainer.append(comp * alpha)
						for element in finalMagContainer: print("{:.4f}".format(element), end=" || ")
						print()
						totalSaltedContainer.append(finalMagContainer)
						self.update_optimizer(optimizer, finalMagContainer)

			
			avg_cost = avg_cost / len(trainloader)
			self.cost_container.append(avg_cost)
			print('cost = {:>.9}'.format(avg_cost), end=" || ")

			with torch.no_grad():
				model.eval()
				correct = 0
				total = 0
				for data in testloader:
					images, labels = data
					images = images.to(device)
					labels = labels.to(device)
					outputs = model(images)

					_, predicted = torch.max(outputs, 1)
					c = (predicted == labels).squeeze()
					for j in range(c.size(dim=0)):
						correct += c[j].item()
						total += 1
				self.val_acc_container.append(correct / total)
				print('Valid Accuracy = {:>.9}%'.format(100 * correct / total))

			if "finalMagContainer" in locals():
				learningStopFlag = True
				for mag in finalMagContainer:
					if mag > 0.01: learningStopFlag = False
				if learningStopFlag: break
			
			scheduler.step()

		print("========== training over! ==========")
		self.save_training_result()
		np.savetxt("./"+ self.prefix_w + '/mag.csv', np.array(totalBaseContainer), delimiter=",")
		np.savetxt("./"+ self.prefix_w + '/salted_mag.csv', np.array(totalSaltedContainer), delimiter=",")

class CIFAR10Trainer(trainer):

	@overrides
	def load_DB(self):
		transform = transforms.Compose(
		[
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

		trainset = torchvision.datasets.CIFAR10(root='DB', train=True,
												download=True, transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
												shuffle=True, num_workers=2)

		testset = torchvision.datasets.CIFAR10(root='DB', train=False,
											download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
												shuffle=True, num_workers=2)
		
		return trainloader, testloader

	@overrides
	def build_num_classes(self):
		self.num_classes = 10 # class number of CIFAR10
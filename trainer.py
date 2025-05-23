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
import matplotlib.pyplot as plt
import math
import json
from overrides import overrides
from safetensors.torch import save_file, safe_open

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
                emb_size=96,
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

	def prepare_save_folder(self, model_name):
		self.model_name = model_name
		self.prefix_w = model_name.upper() + "__" + str(self.rdSeed).zfill(4)

		if not os.path.isdir("./" + self.prefix_w):
			os.mkdir("./" + self.prefix_w)

	def parse_training_args(self, config_file):
		f = open(config_file, 'r')
		self.config = json.load(f)
		self.lr = float(self.config["learning_rate"])
		self.epochs = int(self.config["epochs"])
		self.batch_size = int(self.config["batch_size"])
		self.sampling_step = int(self.config["sampling_step"])
		self.optimizer = self.config["optimizer"] # options : "SGD", "ADAM"
		self.criterion = self.config["criterion"] # options : "CE"
	
	def load_model(self, model_name):
		if model_name.upper() == "UNET":
			self.model_loader = UnetLoader(self.prefix_w, self.num_classes)
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

	def training(self, model_name, rdSeed, config_file_path, is_verbose=False):
		self.set_seed(rdSeed)
		self.prepare_save_folder(model_name)
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
		optimizer = self.load_optimizer(model)
		self.model_loader.save_weight(-1, is_verbose=is_verbose, is_init=True)

		print("========== training start! ==========")

		for epoch in range(self.epochs):
			print("epoch : {:03d} / {:03d} - ".format(epoch + 1, self.epochs), end="")

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

				if i == self.sampling_step: self.model_loader.save_weight(epoch, is_verbose=is_verbose)
			
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

		print("========== training over! ==========")
		self.save_training_result()

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
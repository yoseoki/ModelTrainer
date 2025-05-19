import trainer

myTrainer = trainer.CIFAR10Trainer()


seed = 2802
model_name = "vgg16"
config_file_path = "/home/yoseok/myResearch/run_train/config_{}.json".format(model_name)
is_verbose = False

myTrainer.training(model_name, seed, config_file_path, is_verbose=is_verbose)
"""
# Paper accuracy: 91.25
Num conv: 2
# hyperparam_list = [batch_size, train_valid_split, alpha, epoch]
"""

import torch
import torch.nn as nn
#import torch.nn.parallel
#import torch.backends.cudnn as cudnn
#import torch.distributed as dist
import torch.optim as optim
#import torch.multiprocessing as mp
import torch.utils.data
#import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#import torchvision.models as models

import os, shutil
import numpy as np
import argparse
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as img
#import logging
#from sklearn.model_selection import train_test_split

import model
"""
model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
"""

## THE PARSER: -wd == -w -d; --wd != -w -d
parser = argparse.ArgumentParser(description="CIFAR10 on custom Resnet18")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
	help="number of data loading workers (default: 4)")
parser.add_argument("--epochs", default=500, type=int, metavar="N", ###
	help="no. of total epochs to run (default: 500)")
parser.add_argument("--start_epoch", default="0", type=str, metavar="N", ###
	help="manual epoch number for restarting training (default: 0)")
parser.add_argument("-b", "--batch_size", default=128, type=int, metavar="N",
	help="mini-batch size (default: 128)")
parser.add_argument("--lr", "--learning-rate", default=0.1, type=float, metavar="LR",
	help="init learning rate (default: 0.1)", dest="lr")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", ###
	help="momentum (default: 0.9")
parser.add_argument("--wd", "--weight-decay", default=1e-04, type=float, metavar="W", ###
	help="weight decay (default: 1e-04)", dest="weight_decay")
parser.add_argument("--resume", default="None", type=str, metavar="PATH", ###
	help="path to latest checkpoint (default: none)")
parser.add_argument("--gpu", default=99, type=int,
	help="GPU id to use, check with nvidia-smi")
parser.add_argument("--fname", default="", type=str, metavar="N",
	help="file name of directory (name of sh script)")
parser.add_argument("--step_size", default=100, type=int, metavar="N",
	help="step size for learning rate update (default: 100)")

"""
parser.add_argument("data", default="./data", metavar="DIR", ###ADD DEFAULT
                help="path to dataset")
parser.add_argument("-a", "--arch", metavar="ARCH", default="resnet18",
                choices=model_names,
                help="model architecture: " +
                " | ".join(model_names) +
                " (default: resnet18)")
parser.add_argument("-p", "--print-freq", default=10, type=str, metavar="N",
                help="print frequency default: 10")
"""


def lr_change(count, alpha):
	global optimizer
	alpha=0.1*0.1**count #or should I get the learning rate with param_groups first?
	for param_group in optimizer.param_groups: ##Verified can change lr (print(param_group["lr"]))
		param_group["lr"] = alpha
	print("Learning rate is changed to " + str(alpha) + "!")
	count += 1
	return count, alpha

def train(loader_type, epoch, check, gpuu, device):
	global optimizer, criterion, net, correct, total, class_correct, class_total
	cnt, tstore = 0,0
	if check > 0:
		train_loss, correct, total = 0,0,0
		if check > 1:
			net = net.train()
		else:
			net = net.eval()
	for i, data in enumerate(loader_type, 0):
		images, labels = (data[0].to(device), data[1].to(device)) if gpuu else data
		if check > 0:
			optimizer.zero_grad() #why need in valid mode
		outputs = net(images).to(device) if gpuu else net(images)
		if check > 0:
			loss = criterion(outputs, labels)
			if check > 1:
				loss.backward() #which goes first?
				optimizer.step()
			train_loss += loss.item()
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
		if check < 1:
			zzz, predicted_class = torch.max(outputs, 1)
			c = (predicted_class == labels).squeeze()
			for o in range(4): #why 4?
				label = labels[o]
				class_correct[label] += c[o].item()
				class_total[label] += 1
		else:
			if i % 65 == 64: #65x6x128 (batch size)
				if check == 1:
					f["graph1"][2, epoch] = float('%.3f'%(train_loss/65))
					print("VALID [%d, %5d] loss: %.3f // accuracy: %.3f%%" % (epoch, i+1, train_loss/65, 100*correct/total)) # (0-index, 1-index)
				else:
					print("TRAIN [%d, %5d] loss: %.3f // accuracy: %.3f%%" % (epoch, i+1, train_loss/65, 100*correct/total))
					cnt += 1
					tstore += train_loss
					if cnt == 5:
						f["graph1"][1, epoch] = float('%.3f'%(tstore/325))
						tstore = 0
				train_loss, correct, total = 0,0,0


args = parser.parse_args()

mean, std = (0.4914, 0.48216, 0.44653), (0.24703, 0.24349, 0.26159)
transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])
transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])
trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
train_valid_split = 0.1664 #such that returns 5 train + 1 valid result
valid_num = int(50000*train_valid_split)
data_index = [i for i in range(50000)]
valid_index = np.random.choice(data_index, size=valid_num, replace=False)
validsample = torch.utils.data.sampler.SubsetRandomSampler(valid_index)
trainsample = torch.utils.data.sampler.SubsetRandomSampler(list(set(data_index)-set(valid_index)))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.workers, sampler=trainsample)
validloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.workers, sampler=validsample)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
plateau = 0
epoch_num = args.epochs

net = model.resnet18()

if args.gpu != 99:
	gpuu = True
	device = torch.device("cuda:" + str(args.gpu))
	net.to(device)
else:
	gpuu = False
	device = torch.device("cpu")

criterion = nn.CrossEntropyLoss()
last_acc = 1

if args.resume != "None":
	if os.path.exists(args.fname):
		model = args.resume + args.start_epoch + ".pt"
		checkpoint = torch.load(model)
		net.load_state_dict(checkpoint["net"])
		best_acc = checkpoint["acc"]
		start_epoch = checkpoint["epoch"]
		count = checkpoint["count"] ###
		alpha = checkpoint["alpha"]
		m = checkpoint["momentum"]
		wd = checkpoint["wd"]
		ss = checkpoint["ss"]
		optimizer = optim.SGD(net.parameters(), lr=alpha, momentum=m, weight_decay=wd)
		optimizer.load_state_dict(checkpoint["optimizer"]) #what does this do?
		last_acc = 0
		res = args.resume
if last_acc == 1:
	if os.path.exists(args.fname):
		shutil.rmtree(args.fname)
	os.makedirs(args.fname)
	res = "./" + args.fname + "/checkpoints/"
	last_acc, best_acc, start_epoch, count = 0,0,0,1
	alpha, wd, m, ss = args.lr, args.weight_decay, args.momentum, args.step_size
	optimizer = optim.SGD(net.parameters(), lr=alpha, momentum=m, weight_decay=wd)
	print("Trained from scratch: " + "// lr " + str(alpha) + " // epoch_num " + str(epoch_num) +" // wd " + str(wd) + " // momentum " + str(m))
else:
	print("Loaded from " + model + " : // best_acc " + str(best_acc) + " // lr " + str(alpha) + " // epoch_num " + str(epoch_num) + " // wd " + str(wd) + " // momentum " + str(m))
	print("\n" + "\n")
	


with h5py.File(args.fname + ".hdf5", "a") as f:
	ks = list(f.keys())
	arra = [-1.123 for i in range(epoch_num)]
	if len(ks) == 0:
		f.create_dataset("graph1", data=np.array([np.array([i for i in range(epoch_num)]), arra, arra]))
		f.create_dataset("graph2", data=np.array([np.array([i for i in range(epoch_num)]), arra]))
	else:
		if "graph1" not in ks:
			f.create_dataset("graph1", data=np.array([np.array([i for i in range(epoch_num)]), arra, arra]))
		if "graph2" not in ks:
			f.create_dataset("graph2", data=np.array([np.array([i for i in range(epoch_num)]), arra]))

	## TRAIN VALID TEST
	for epoch in range(start_epoch, start_epoch+epoch_num):
		net = net.train()
		if not os.path.exists(res):
			os.makedirs(res)
		if epoch <= (ss*3-1) and epoch % ss == ss-1: ##TUNEE: rn epoch_num=200(500), step size=60
			count, alpha = lr_change(count, alpha)
		if epoch % 100 == 99:
			x1 = f["graph1"][0][:epoch+1]
			x2 = f["graph2"][0][:epoch+1]
			t_l, v_l, t_a = f["graph1"][1][:epoch+1], f["graph1"][2][:epoch+1], f["graph2"][1][:epoch+1]
			plt.plot(x1, t_l, label="train loss")
			plt.plot(x2, v_l, label="valid loss")
			plt.xlabel("Number of epochs")
			plt.ylabel("Loss")
			plt.legend(loc="best")
			plt.title(label="Train Valid Loss: Epoch " + str(epoch))
			plt.savefig(res + args.fname + "_" + str(epoch) + "_1.png")
			plt.clf()
			plt.plot(x2, t_a, label="test accuracy")
			plt.xlabel("Number of epochs")
			plt.ylabel("Accuracy")
			plt.legend(loc="best")
			plt.title(label="Test Accuracy: Epoch " + str(epoch))
			plt.savefig(res + args.fname + "_" + str(epoch) + "_2.png")
			plt.clf()
		correct, total, class_correct, class_total = 0,0, 1, 1
		net = net.train()
		train(trainloader, epoch, 2, gpuu, device)
		net = net.eval()
		train(validloader, epoch, 1, gpuu, device)
		correct, total, class_correct, class_total = 0, 0, [0 for i in range(10)], [0 for i in range(10)]
		with torch.no_grad(): #no need backprop
			train(testloader, epoch, 0, gpuu, device)
		acc = 100*correct/total
		foo = str(epoch)
		print("Epoch #" + foo, "TEST ACCURACY:", acc)
		f["graph2"][1, epoch] = acc
		
		state = {
			"net": net.state_dict(),
			"acc": acc,
			"epoch": epoch,
			"count": count,
			"optimizer": optimizer.state_dict(),
			"wd": wd,
			"momentum": m,
			"alpha": alpha,
			"ss": ss
			}
		torch.save(state, res + foo + ".pt")

		if acc > best_acc:
			print("Checkpoint!")
			torch.save(state, res + "ckpt.pt")
			best_acc = acc
		for i in range(10):
			print("Epoch #" + foo, "accuracy of ", classes[i], 100*class_correct[i]/class_total[i])
		print("\n")
	es = str(epoch)
	x1 = f["graph1"][0] #epochs
	x2 = f["graph2"][0] #epochs
	train_loss, valid_loss = f["graph1"][1], f["graph1"][2]
	test_acc = f["graph2"][1]
	
	#graph 1: over/under-fitting
	plt.plot(x1, train_loss, label="train loss")
	plt.plot(x1, valid_loss, label="valid loss")
	plt.xlabel("Number of epochs")
	plt.ylabel("Loss")
	plt.legend(loc="best")
	plt.title(label="Final Train Valid Loss: Epoch " + es)
	plt.savefig(res + args.fname + "_".join(["", es, "1", "final"]) + ".png")

	plt.clf()

	#graph 2: test accuracy
	plt.plot(x2, test_acc, label="test accuracy")
	plt.xlabel("Number of epochs")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.title(label="Final Test Accuracy: Epoch " + es)
	plt.savefig(res + args.fname + "_".join(["", es, "2", "final"]) + ".png")

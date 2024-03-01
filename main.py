from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from train import train
from model.vgg import VggNet
from model.alexnet import AlexNet
from model.lenet import LeNet
from model.densenet import DenseNet
from model.nin import NiNNet
from model.googlenet import GoogLeNet
from model.resnet import ResNet
import torch


import argparse 

# 创建一个参数解析实例
parser = argparse.ArgumentParser(description='Parameters') 

# 添加参数解析
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--model_name', type=str, choices=['VggNet', 'AlexNet', 'LeNet', 'DenseNet', 'NiNNet', 'GoogLeNet', 'ResNet'] ,default='LeNet')
args = parser.parse_args()


batch_size = args.batch_size
lr = args.learning_rate

transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),  # Convert the PIL Image to Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化；0.1307为均值，0.3081为标准差
])

# Download and load the training data
train_set = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Download and load the test data
test_set = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)



use_cuda = torch.cuda.is_available()

if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


net = eval(args.model_name+"()")
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()

train(net, loss, optimizer, train_loader, test_loader, num_epochs=args.epochs, device=device)

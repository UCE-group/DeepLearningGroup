import torchvision
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
import time

data_tf = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize([0.5], [0.5])]
)
train_set = torchvision.datasets.MNIST('./data', train=True, transform=data_tf, download=True)
test_set = torchvision.datasets.MNIST('./data', train=False, transform=data_tf,download=True)

train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

class my_mnist(nn.Module):
    def __init__(self):
        super(my_mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Linear(32*7*7, 10)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        out = self.layer3(x)
        return out
net = my_mnist()
net.cuda()

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), 1e-2)

for e in range(500):
    start = time.time()

    train_loss = 0
    train_acc = 0
    for x, y in train_data:
        x_input = Variable(x).cuda()
        y_target = Variable(y).cuda()

        output = net(x_input)
        loss = loss_func(output, y_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pred = torch.max(output, 1)[1].cuda()
        num_correct = (pred == y_target).sum()
        train_acc += num_correct.item()

    train_loss /= len(train_data)*128
    train_acc /= len(train_data)*128

    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}'.format(e, train_loss, train_acc))
    torch.save(net.state_dict(), 'pkl/mnist.pkl')
    print('Have costed {}'.format(time.time()-start))
def test():
    net = torch.load('pth/499net.pth').cuda()
    net.eval()

    test_acc = 0
    for img, label in test_data:
        test_img = Variable(img).cuda()
        label = Variable(label).cuda()

        test_out = net(test_img)
        pred = torch.max(test_out,1)[1].cuda()
        num_correct = (pred == label).sum()
        test_acc += num_correct.item()

    test_acc /= len(test_data)*128

    print('Test Acc: {:.6f}'.format(test_acc))

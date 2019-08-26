#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a3_neural_network.py
@Time: 2019-10-17 17:29
@Last_update: 2019-10-17 17:29
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        print('here')
        print(x.shape)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        print(x.shape)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        print(x.shape)
        x = x.view(-1, self.num_flat_featurs(x))
        print(x.shape)
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = F.relu(self.fc2(x))
        print(x.shape)
        x = self.fc3(x)
        print(x.shape)

        return x

    def num_flat_featurs(self, x):
        size = x.size()[1:]
        num_featurs = 1
        for s in size:
            num_featurs *= s

        return num_featurs

if __name__ == '__main__':
    net = Net()
    print(net)
    params = list(net.parameters())
    print(len(params))
    print(params[0].shape)
    for i in net.parameters():
        print(i)

    optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer.zero_grad()
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    target = torch.randn(10)
    target = target.view(1, -1)
    criterion = nn.MSELoss()

    loss = criterion(out, target)
    print(loss)
    net.zero_grad()
    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)
    loss.backward()

    # print(loss.grad_fn)
    # print(loss.grad_fn.next_functions[0][0])
    # print(loss.grad_fn.next_functions[0][0].next_functions[0][0])
    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)
    optimizer.step()

    # learning_rate = 0.01
    # for f in net.parameters():
    #     f.data.sub_(f.grad.data * learning_rate)
    #     print(f)
    #     print(f.data)
    #     print(f.grad)
    #     print(f.grad.data)
    #     exit()


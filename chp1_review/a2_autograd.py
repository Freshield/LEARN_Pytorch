#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a2_autograd.py
@Time: 2019-10-17 17:04
@Last_update: 2019-10-17 17:04
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
import numpy as np

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
b = (a * a).sum()
print(b)
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b)

out.backward()
print(x.grad)

x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)


print()
y_pred = torch.from_numpy(np.arange(0.1, 1, 0.1).reshape((3,3)))
y_pred.requires_grad_(True)
print(y_pred)
y_true = torch.tensor([0,0,0,0,1,1,0,1,1], dtype=torch.double, requires_grad=True).reshape((3,3))
print(y_true)

dice_loss = 1 - (2 * (y_pred * y_true).sum() / (y_pred.sum() + y_true.sum()))
print(dice_loss)

dice_loss.backward()
print(y_pred.grad)
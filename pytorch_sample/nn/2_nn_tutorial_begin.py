# -*- coding: utf-8 -*-
"""
What is `torch.nn` *really*?
============================
by Jeremy Howard, `fast.ai <https://www.fast.ai>`_. Thanks to Rachel Thomas and Francisco Ingham.
"""
###############################################################################
# We recommend running this tutorial as a notebook, not a script. To download the notebook (.ipynb) file,
# click the link at the top of the page.
#
# PyTorch provides the elegantly designed modules and classes `torch.nn <https://pytorch.org/docs/stable/nn.html>`_ ,
# `torch.optim <https://pytorch.org/docs/stable/optim.html>`_ ,
# `Dataset <https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset>`_ ,
# and `DataLoader <https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader>`_
# to help you create and train neural networks.
# In order to fully utilize their power and customize
# them for your problem, you need to really understand exactly what they're
# doing. To develop this understanding, we will first train basic neural net
# on the MNIST data set without using any features from these models; we will
# initially only use the most basic PyTorch tensor functionality. Then, we will
# incrementally add one feature from ``torch.nn``, ``torch.optim``, ``Dataset``, or
# ``DataLoader`` at a time, showing exactly what each piece does, and how it
# works to make the code either more concise, or more flexible.
#
# **This tutorial assumes you already have PyTorch installed, and are familiar
# with the basics of tensor operations.** (If you're familiar with Numpy array
# operations, you'll find the PyTorch tensor operations used here nearly identical).
#
# MNIST data setup
# ----------------
#
# We will use the classic `MNIST <http://deeplearning.net/data/mnist/>`_ dataset,
# which consists of black-and-white images of hand-drawn digits (between 0 and 9).
#
# We will use `pathlib <https://docs.python.org/3/library/pathlib.html>`_
# for dealing with paths (part of the Python 3 standard library), and will
# download the dataset using
# `requests <http://docs.python-requests.org/en/master/>`_. We will only
# import modules when we use them, so you can see exactly what's being
# used at each point.

from pathlib import Path
import sys
import requests

# 创建本地数据库路径
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
PATH.mkdir(parents=True, exist_ok=True)

# 下载远程数据源
URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

###############################################################################
# This dataset is in numpy array format, and has been stored using pickle,
# a python-specific format for serializing data.

# 打开本地数据源，分为训练数据源和测试数据源
import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

###############################################################################
# Each image is 28 x 28, and is being stored as a flattened row of length
# 784 (=28x28). Let's take a look at one; we need to reshape it to 2d
# first.

from matplotlib import pyplot
import numpy as np

# x_train.shape: (50000, 784) 50000张图片(28*28)
# y_train 是多对应x_train 的标签值
# 每个图像为28 x 28，并存储为长度为784 = 28x28的扁平行。 让我们来看一个； 我们需要先将其重塑为 2d。
pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")

###############################################################################
# PyTorch uses ``torch.tensor``, rather than numpy arrays, so we need to
# convert our data.

import torch

# numpy 转换成 tensor
# PyTorch 使用torch.tensor而不是 numpy 数组，因此我们需要转换数据。
x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
n, c = x_train.shape

# x_train.shape: (50000, 784) 50000张图片(28*28)
print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("y_train.value:",y_train)
print(y_train.min(), y_train.max())


###############################################################################
# Neural net from scratch (no torch.nn)
# ---------------------------------------------
#
# Let's first create a model using nothing but PyTorch tensor operations. We're assuming
# you're already familiar with the basics of neural networks. (If you're not, you can
# learn them at `course.fast.ai <https://course.fast.ai>`_).
#
# PyTorch provides methods to create random or zero-filled tensors, which we will
# use to create our weights and bias for a simple linear model. These are just regular
# tensors, with one very special addition: we tell PyTorch that they require a
# gradient. This causes PyTorch to record all of the operations done on the tensor,
# so that it can calculate the gradient during back-propagation *automatically*!
#
# For the weights, we set ``requires_grad`` **after** the initialization, since we
# don't want that step included in the gradient. (Note that a trailing ``_`` in
# PyTorch signifies that the operation is performed in-place.)
#
# .. note:: We are initializing the weights here with
#    `Xavier initialisation <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
#    (by multiplying with 1/sqrt(n)).

import math

weights = torch.randn(784, 10) / math.sqrt(784)

# weights.requires_grad_() 默认值为true,设置为需要自动求导，避免上面公式加入自动求导
weights.requires_grad_()

bias = torch.zeros(10, requires_grad=True)

###############################################################################
# Thanks to PyTorch's ability to calculate gradients automatically, we can
# use any standard Python function (or callable object) as a model! So
# let's just write a plain matrix multiplication and broadcasted addition
# to create a simple linear model. We also need an activation function, so
# we'll write `log_softmax` and use it. Remember: although PyTorch
# provides lots of pre-written loss functions, activation functions, and
# so forth, you can easily write your own using plain python. PyTorch will
# even create fast GPU or vectorized CPU code for your function
# automatically.

# 手写 softmax 函数
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

# xb 和weight 进行点积
# 等价于 matmul 进行向量矩阵点积
def model(xb):
    return log_softmax(xb @ weights + bias)

bs = 64  # batch size

# x_train.shape: (50000, 784) 50000张图片(28*28)
# 每次从抽取64张图片进行训练
xb = x_train[0:bs]  # a mini-batch from x

# 进行训练模型预测
preds = model(xb)  # predictions

# torch.Size([64, 10])
# init weight & bias shape is:
#   bias = torch.zeros(10, requires_grad=True)
#   weights = torch.randn(784, 10) / math.sqrt(784)
print("prediction model data:",preds[0], preds.shape)

# 定义损失函数loss_function
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()
loss_func = nll

# 进行损失值计算
yb = y_train[0:bs]
print(loss_func(preds, yb))

# 计算精确度
# 我们还实现一个函数来计算模型的准确率。 对于每个预测，如果具有最大值的索引与目标值匹配，则该预测是正确的。
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

# 第一次打印计算精度
print("first training:",accuracy(preds, yb))

# 现在，我们可以运行一个训练循环
lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for
for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

print("epochs training:",loss_func(model(xb), yb), accuracy(model(xb), yb))
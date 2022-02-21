from pathlib import Path
import sys
import requests
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# loss_function
loss_func = F.cross_entropy

# MNIST_Logistic
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)


# accuracy
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

# region 获取远程数据集并读取
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

# endregion


bs = 64  # batch size
lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

# x_train.shape: (50000, 784) 50000张图片(28*28)
# 每次从抽取64张图片进行训练
xb = x_train[0:bs]  # a mini-batch from x

##################
# 批量训练
##################
model, opt = get_model()

# 加载测试数据集
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

# 加载验证数据集
# 无论我们是否打乱验证集，验证损失都是相同的。 由于打乱需要花费更多时间，因此打乱验证数据没有任何意义。
# 我们将验证集的批量大小设为训练集的两倍。 这是因为验证集不需要反向传播，因此占用的内存更少（不需要存储梯度）。 
# 我们利用这一优势来使用更大的批量，并更快地计算损失。
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)


# 手动循环加载
# for epoch in range(epochs):
#     for i in range((n - 1) // bs + 1):

#         # 手动批量获取
#         # start_i = i * bs
#         # end_i = start_i + bs
#         # xb = x_train[start_i:end_i]
#         # yb = y_train[start_i:end_i]

#         # 通过Dataset 数据集批量获取
#         xb,yb = train_ds[i*bs : i*bs+bs]

#         pred = model(xb)
#         loss = loss_func(pred, yb)

#         loss.backward()
#         opt.step()
#         opt.zero_grad()

# DataLoader 自动遍历
for epoch in range(epochs):

    # 进行模型训练模式
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    # 进行验证评估模式
    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))

print("epochs training:",loss_func(model(xb), yb), accuracy(model(xb), yb))

##################################################################3
# 创建fit()和get_data()
# 将训练过程和验证过程集成到一个函数进行处理
import numpy as np
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

# train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
# model, opt = get_model()
# fit(epochs, model, loss_func, opt, train_dl, valid_dl)
import numpy as np
import torch
from torch.utils import data
from torch import nn

def synthetic_data(w, b, num_examples):  #@save
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))
    
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 变量这个数据集
next(iter(data_iter))

# nn是神经网络的缩写
# Linear(形状特征，输出)
net = nn.Sequential(nn.Linear(2, 1))

# 手动初始化，权重和偏置
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.03)


num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        l.backward()
        trainer.step()
        trainer.zero_grad()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
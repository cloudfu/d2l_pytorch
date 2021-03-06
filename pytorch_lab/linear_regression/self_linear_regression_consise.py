import torch
import random
from torch.utils import data
from torch import nn
from d2l import torch as d2l


# 生成数据模型
def generate_train_data(w,b,count):
    """[生成数据模型]
    Args:
        w (int): 权重
        b (int): 偏移量
        count (int): 需要生成测试数据的数量
    Returns:
        [Tensor]: 返回x和y比对数据
    """
    # x = torch.normal(mean=0, std=torch.linspace(0, 1,count))
    features = torch.normal(0, 1, (count ,1))

    # 模拟X和Y之间的关系
    labels = features * w + b

    labels += torch.normal(0, 0.4, labels.shape)
    return features,labels

# def iterate_data(batch_size, x, y):
#     """小批量获取测试数据，并对于提供测试数据进行打散处理
#     Args:
#         batch_size ([int]): [批量数据大小]
#         x ([Tensor]): [x 维度数据]
#         y ([Tensor]): [y 维度数据]
#     """

#     train_size = len(x)
#     # 获取Tensor索引
#     index = list(range(train_size))
#     # 进行索引打散，后续通过索引进行随机获取训练数据
#     random.shuffle(index)

#     for i in range(0, train_size, batch_size):
#         batch_index = index[i:min(i+batch_size,train_size)]
#         yield x[batch_index],y[batch_index]


# 批量加载数据
def load_data(data_arrays,batch_size,is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_train)


# # 预测函数,predict_fun 线性回归
# def linear_regression(x,w,b):
#     return x * w + b

# # 平方差损失计算
# def loss_squared(true_y,predict_y):
#     return (true_y - predict_y)**2/2

# # SDG 随机梯度下降
# def sgd(params,lr,batch_size):
#     with torch.no_grad():
#         for param in params:
#             param -= lr * param.grad / batch_size 
#             param.grad.zero_()

# 需要生成训练数据的数量
train_data_size = 1000
# 最小批次的训练量
batch_size = 10

# 真实权重
true_weight = 3.4
# 真实偏置
true_bias = 7

# 获取测试数据集
features,labels = generate_train_data(true_weight,true_bias,train_data_size)
data_iter = load_data((features, labels), batch_size)

# 查看生成数据模型
# d2l.set_figsize()
# d2l.plt.scatter(x, y, 1)
# d2l.plt.show()

# 迭代出需要批次处理数据
# for x,y in iterate_data(batch_size,x,y):
#     print("x:",x,"y",y)


# 准备初始化数据
# w = torch.ones(1, requires_grad=True)
# b = torch.ones(1, requires_grad=True)

# 学习率
lr = 0.03

# 设定神经网络、损失函数，梯度下降
net = nn.Sequential(nn.Linear(1,1))
# 手动初始化，权重和偏置
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(),lr)


# 测试循环次数
num_epochs = 3
# 开始进行梯度下降计算
for epoch in range(num_epochs):
    for features,labels in data_iter:
        # 进行线性预测
        predict_y = net(features)

        # 每次训练不停的调整weight/bias 两个参数
        print("w",net[0].weight.data,"b",net[0].bias.data)

        # 损失计算
        l = loss(predict_y,labels)

        # 梯度归零，这对每个parameters
        trainer.zero_grad()

        l.backward()
        trainer.step()

        # # 进行线性预测
        # predict_y = net(features,w,b)

        # # 损失计算
        # l = loss(labels, predict_y)
        # l.sum().backward()

        # # 进行梯度下降
        # sgd([w,b],lr,batch_size)

    print(f'epoch {epoch + 1}, loss {l:f}')

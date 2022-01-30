import torch
import random
from d2l import torch as d2l

# 需要生成训练数据的数量
train_data_size = 1000
# 最小批次的训练量
batch_size = 10
# 权重
weight = 3.4
# 偏移量
b = 7
# 学习率
lr = 0.03
# 测试循环次数
num_epochs = 3

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
    x = torch.normal(0, 1, (count ,1))
    y = x * w + b
    y += torch.normal(0, 1, y.shape)
    return x,y

def iterate_data(batch_size, x, y):
    """小批量获取测试数据，并对于提供测试数据进行打散处理
    Args:
        batch_size ([int]): [批量数据大小]
        x ([Tensor]): [x 维度数据]
        y ([Tensor]): [y 维度数据]
    """

    train_size = len(x)
    # 获取Tensor索引
    index = list(range(train_size))
    # 进行索引打散，后续通过索引进行随机获取训练数据
    random.shuffle(index)

    for i in range(0, train_size, batch_size):
        batch_index = index[i:min(i+batch_size,train_size)]
        yield x[batch_index],y[batch_index]

# 预测函数,predict_fun 线性回归
def linear_regression(x,w,b):
    return x * w + b

# 平方差损失计算
def loss_squared(true_y,predict_y):
    return (true_y - predict_y)**2/2

# SDG 随机梯度下降
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size 
            param.grad.zero_()


# 获取测试数据集
x,y = generate_train_data(weight,b,train_data_size)

# 查看生成数据模型
# d2l.set_figsize()
# d2l.plt.scatter(x, y, 1)
# d2l.plt.show()

# 迭代出需要批次处理数据
# for x,y in iterate_data(batch_size,x,y):
#     print("x:",x,"y",y)


# 准备初始化数据
w = torch.ones(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# 通过方法复制，方便后续进行维护调整
net = linear_regression
loss = loss_squared

# 开始进行梯度下降计算
for epoch in range(num_epochs):
    for x,y in iterate_data(batch_size, x, y):
        # 进行线性预测
        predict_y = net(x,w,b)

        # 损失计算
        l = loss(y, predict_y)
        print(float(l.mean()))
        l.sum().backward()

        # 进行梯度下降
        sgd([w,b],lr,batch_size)

        # 打印每次批次之后的递进值
    with torch.no_grad():
        train_loss = loss(y,net(x,w,b))
        print("epoch", epoch + 1, "loss", float(train_loss.mean()))

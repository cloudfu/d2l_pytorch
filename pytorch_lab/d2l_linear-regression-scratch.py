
import random
import torch
from d2l import torch as d2l


def synthetic_data(w, b, num_examples):  #@save
    # w为权重 w = torch.tensor([2, -3.4])
    # b为偏移量 b = true_b = 4.2
    # num_examples = 1000

    # torch.normal(means, std, out=None)
    #   means (Tensor) – 均值，可以接收一个向量，或者一个标量
    #   std (Tensor) – 标准差，可以接收一个向量，或者一个标量，means,std 会进行广播
    #   out (Tensor) – 可选的输出张量

    # 生成 平局数为0，标准差为1 (1000,2)的样例数据,
    x = torch.normal(0, 1, (num_examples, len(w)))
    # X = (1000,2)

    # Y:(1000,2) * [2, -3.4] 
    # 降维由 (1000,2) 转变 (1000)，其中乘以 [2, -3.4] 造成 维度交叉[0]和[1]
    y = torch.matmul(x, w) + b
    # y:(1000)

    y += torch.normal(0, 0.01, y.shape)
    
    # y.reshape(-1,1) = (1000,1)
    return x, y.reshape((-1, 1))

# w为权重
true_w = torch.tensor([2, -3.4])
# b为偏移量
true_b = 4.2
# y(1,1000)

# 生成y=Xw+b+噪声
# feature=(1000, 2) 每一行都包含一个二维数据样本
# lables = (1000, 1) 每一行都包含一维标签值（一个标量）
features, labels = synthetic_data(true_w, true_b, 1000)
print(labels.shape)

"""
查看当前生成的测试数据
"""
# d2l.set_figsize()
# # scatter(x,y,point_size)
# d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 3)
# d2l.plt.show()

def data_iter(batch_size, features, labels):
    """小批量枚举测试数据集

    Args:
        batch_size ([type]): [每批次数量大小]
        features ([type]): [description]
        labels ([type]): [description]

    Yields:
        [type]: [description]
    """
    # num_examples=1000
    num_examples = len(features)

    # 转换成1000数组
    indices = list(range(num_examples))

    # 打散数组
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        
        # min(i + batch_size, num_examples) 有可能batch 叠加之后超出num_examples
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])

        yield features[batch_indices], labels[batch_indices]



batch_size = 10
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

lr = 0.03
num_epochs = 3

# 通过方法复制，方便后续进行维护调整
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):

        # print("w:",w.data,"b:",b.data )
        # net 是激活函数
        # X:(10,2)
        # w权重:(2,1)
        # b偏移量:[0.]
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()

        #随机梯度下降法，调整w权重和b偏移量
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        print("loss:", float(l.mean()))

    # with 是在一个批次训练完成之后进行比较loss
    with torch.no_grad():
        # print("w:",w.data,"b:",b.data )
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
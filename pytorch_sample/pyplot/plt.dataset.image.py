import os
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from PIL import Image
from torchvision import transforms

# 可以提高显示清晰度
# d2l.use_svg_display()

print(os.getcwd())

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0到1之间
trans = transforms.ToTensor()
# debug 运行为当前项目root目录
mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=False)
mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=False)
# run
# mnist_train = torchvision.datasets.FashionMNIST(root="./d2l_pytorch/data", train=True, transform=trans, download=False)
# mnist_test = torchvision.datasets.FashionMNIST(root="./d2l_pytorch/data", train=False, transform=trans, download=False)

print("mnist_train", len(mnist_train))
print("mnist_test", len(mnist_test))

# torch.Size([1, 28, 28])
# 黑白1维图片
print(mnist_train[0][0].shape)

def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)

    # 设定当前图标可支持多少图形容器
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
# 、
    # 压平程一维容器 从 (0,)(0,1)(1,0)(1,1) -> (1,2,3,4)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


# 获取批量训练数据
# mnist_train[0][0].shape = torch.Size([1, 28, 28]) 
# 一共10000组训练数据，每组数据包括18个图像，28*28个像素
# X:为18张图像，X.shape = (18,1,28,28)
# y:为Lable索引
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))

# 获取y对应的标签名称
titles=get_fashion_mnist_labels(y)

# 
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
d2l.plt.show()

batch_size = 256

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 1

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True)

timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')


def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    # run
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=False)
    # debug
    # mnist_train = torchvision.datasets.FashionMNIST(root="./d2l_pytorch/data", train=True, transform=trans, download=False)
    # mnist_test = torchvision.datasets.FashionMNIST(root="./d2l_pytorch/data", train=False, transform=trans, download=False)


    # 需要删除 num_workers=get_dataloader_workers() 入参，不然Windows平台运行会有错误
    return (data.DataLoader(mnist_train, batch_size, shuffle=True),
            data.DataLoader(mnist_test, batch_size, shuffle=False))

# commit
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
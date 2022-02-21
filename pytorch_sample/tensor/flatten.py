import torch
x=torch.randn(2,4,2)
# print(x)
 
# ---------------------
# torch.flatten 方法解释
# ---------------------

# 全维度平坦化
# z=torch.flatten(x)
# print(z)
# print(z.shape)
 
# 从第几维度开始平坦化，维度从0开始
# w=torch.flatten(x,2)
# print(w)

# torch.flatten(x,0,1)代表在第一维和第二维之间平坦化
# 第一维长度2，第二维长度为4，平坦化后长度为2*4
# w=torch.flatten(x,0,1) 
# print(w.shape)
# # torch.Size([8, 2])
# print(w)

# ---------------------
# torch.nn.Flatten 方法解释
# ---------------------
x = torch.ones(2, 2, 2, 2)

# 开始维度默认为 1。因为其被用在神经网络中，输入为一批数据
# default：F = torch.nn.Flatten(1) = torch.Size([2, 8])
F = torch.nn.Flatten(0,1)
y = F(x)
print(y)
print(y.shape)




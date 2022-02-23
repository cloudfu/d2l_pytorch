import os
from torch.utils.tensorboard import SummaryWriter


print(os.getcwd())
if __name__ == '__main__':
    writer = SummaryWriter('d:/log/scalar_example')
    for i in range(10):
        writer.add_scalar('quadratic', i**2, global_step=i)
        writer.add_scalar('exponential', 2**i, global_step=i)

    writer.close()


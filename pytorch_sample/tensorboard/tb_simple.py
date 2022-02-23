import os
from torch.utils.tensorboard import SummaryWriter


path_ = os.getcwd()
print(path_)

# if __name__ == '__main__':
#     writer = SummaryWriter('./log/scalar_example')
#     for i in range(10):
#         writer.add_scalar('quadratic', i**2, global_step=i)
#         writer.add_scalar('exponential', 2**i, global_step=i)

#     writer.close()


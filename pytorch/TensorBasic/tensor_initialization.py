import torch

# CUDA is to run on GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# common initialization

x = torch.empty(size=(2, 3))
# print(x)
x = torch.zeros((3, 3))
# print(x)
x = torch.rand((3, 3))  # 0 to 1
# print(x)
x = torch.ones((3, 3))
# print(x)
x = torch.eye(4, 4)  # identity matrix
# print(x)
x = torch.arange(start=0, end=5, step=2)
# print(x)
x = torch.linspace(start=0.1, end=1, steps=20)
# print(x)
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
# print(x)
x = torch.empty(size=(1, 5)).uniform_(0, 1)
# print(x)
x = torch.diag(torch.ones(3))  # identity matrix of 3x3
# print(x)

# Initializing and converting to other types (int, float, double)
tensor = torch.arange(4)
# print(tensor.bool())
# print(tensor.short())  # int16
# print(tensor.long())
# print(tensor.half())  # float 16
# print(tensor.double())  # float 64

# array to tensor and viceversa
import numpy as np

nparray = np.zeros((5, 5))
print(nparray)
tensor = torch.from_numpy(nparray)
print(tensor)
nparrayback = tensor.numpy()
print(nparrayback)

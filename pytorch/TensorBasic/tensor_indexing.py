import torch

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

# print(x[0].shape)  # x[0,:]
# print(x[:, 0].shape)

# print(x[2, 0:10])

# fancy indexing
x = torch.arange(10)
indices = [2, 5, 8]
# print(x[indices])

x = torch.rand((3, 5))
# print(x)
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
# print(x[rows, cols]) #has shape of 2

# advanced indexing
x = torch.arange(10)
# print(x[(x < 2) | (x > 8)])
# print(x[x.remainder(2) == 0]) #all even elements

# useful operations
print(torch.where(x > 5, x, x * 2))
print(torch.tensor([0, 0, 1, 2, 3, 4, 3]).unique())
print(x.ndimension())
print(x.numel())  # number of elements


import torch

x = torch.arange(16)

# x = x.view(4, 4)
x = x.reshape(4, 4)

y = x.t()  # transpose
# print(y.contiguous().view(16))

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
# print(torch.cat((x1, x2), dim = 0).shape) # (4, 5)
# print(torch.cat((x1, x2), dim = 1).shape) # (2, 10)

z = x1.view(-1)
# print(z.shape)  # it flattens

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
# print(z.shape) #(64, 10)

z = x.permute(0, 2, 1)  # permuting

x = torch.arange(10)  # [10]
# print(x.unsqueeze(0).shape)  # [1, 10]
# print(x.unsqueeze(1).shape)  # [10, 1]

x = torch.arange(10).unsqueeze(0).unsqueeze(1)
#print(x.shape)  # ([1, 1, 10]
z = x.squeeze(1)
#print(z.shape)  # [1, 10]

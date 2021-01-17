import torch

x1 = torch.tensor([1, 2, 3])
x2 = torch.tensor([4, 5, 6])

# addition
a1 = torch.empty(3)
torch.add(x1, x2, out=a1)

a2 = torch.add(x1, x2)
a = x1 + x2

# subtraction

s = x1 - x2

# Division
d = torch.true_divide(x1, x2)  # element wise division

# inplace operations
t = torch.zeros(3)
t.add_(x1)
t += x1
# print(t)

# exponentiation
z = x1.pow(2)  # element wise
z = x1 ** 2

# simple comparison
z = x1 > 0
z = x1 < 0
# print(z)

# matrix multiplication
x = torch.rand((2, 5))
y = torch.rand((5, 3))
z = torch.mm(x, y)
# print(z)
z = x.mm(y)

# matrix exponential
matrixExpo = torch.rand(5, 5)
matrixExpo.matrix_power(3)

# element wise multiplication
z = x1 * x2

# dot product
z = torch.dot(x1, x2)
# print(z)

# batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out = torch.bmm(tensor1, tensor2)  # (batch, n, p)

# example of broadcasting
x = torch.rand((5, 5))
y = torch.rand((1, 5))

z = x - y
# print(z)
z = x ** y
# print(z)

# other useful stuffs
sum = torch.sum(x1, dim=0)
# print(sum)
values, indices = torch.max(x1, dim=0)  # x1.max(dim=0)
values, indices = torch.min(x1, dim=0)
absX = torch.abs(x1)
z = torch.argmax(x1, dim=0)
mean = torch.mean(x1.float(), dim=0)
z = torch.eq(x1, x2)  # compare each element
sortedY, indices = torch.sort(x2, dim=0, descending=False)

z = torch.clamp(x1, min=0, max=10)  # less than 0 n more than 10 is set to 0

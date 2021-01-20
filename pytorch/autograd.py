import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
# print(y)
z = y * y * 2
# z = z.mean()
# print(z)

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v)
# print(x.grad)

# to remove gradient
# x.requires_grad = False
# x.detach()
# with torch.no_grad()

# x.requires_grad_(False) #trailing underscore modifes the inplace value
# print(x)
# y = x.detach()
# print(y)

with torch.no_grad():
    y = x + 2
    # print(y)

weights = torch.ones(4, requires_grad=True)

optimizer = torch.optim.sgd(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()

for epoch in range(3):
    model_output = (weights * 3).sum()

    model_output.backward()

    print(weights.grad)

    weights.grad.zero_()  # reset the grad

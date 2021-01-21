# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
# - forward pass:  compute prediction
# - backward pass: gradient
# - update weights


import torch
import torch.nn as nn
import torch.optim as optim

# function

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape
# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
# def forward(x):
#     return w * x

input_size = n_features
output_size = n_features


# model = nn.Linear(input_size, output_size)


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define Layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


model = LinearRegression(input_size, output_size)

# print(f'Prediction before training: f(5) = {forward(5):.3f}')
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 100  # torch requires more iterations as backward is not the same as numerical calculation
loss = nn.MSELoss()
# optimizer = optim.SGD([w], lr=learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_hat = model(X)

    # loss
    l = loss(Y, y_hat)

    # gradient = backward pass
    l.backward()  # gradient of loss wrt w

    # update weights
    optimizer.step()

    # zero gradients
    # w.grad.zero_()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss={l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')

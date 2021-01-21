import torch

# function

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


# model prediction
def forward(x):
    return w * x


# loss = Mean Squared Error
def loss(y, y_hat):
    return ((y_hat - y) ** 2).mean()


print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 100 # torch requires more iterations as backward is not the same as numerical calculation

for epoch in range(n_iters):
    # prediction = forward pass
    y_hat = forward(X)

    # loss
    l = loss(Y, y_hat)

    # gradient = backward pass
    l.backward() # gradient of loss wrt w

    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

    # zero gradients
    w.grad.zero_()

    if epoch % 5 == 0:
        print(f'epoch {epoch + 1}: w = {w:.3f}, loss={l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')

import numpy as np

# function

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0


# model prediction
def forward(x):
    return w * x


# loss = Mean Squared Error
def loss(y, y_hat):
    return ((y_hat - y) ** 2).mean()


# gradient
# MSE + 1/N *(w*x - y)**2
# dj/dw = 1/N* 2x * (w*x - y)

def gradient(x, y, y_hat):
    return np.dot(2 * x, y_hat - y).mean()


print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    # prediction = forward pass
    y_hat = forward(X)

    # loss
    l = loss(Y, y_hat)

    # gradient
    dw = gradient(X, Y, y_hat)

    # update weights
    w -= learning_rate * dw

    if epoch % 2 == 0:
        print(f'epoch {epoch + 1}: w = {w:.3f}, loss={l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')

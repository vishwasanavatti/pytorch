# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
# - forward pass:  compute prediction
# - backward pass: gradient
# - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) preparing data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
# print(n_samples, n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale
sc = StandardScaler()  # 0 mean n unit variance
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)  # column vector
y_test = y_test.view(y_test.shape[0], 1)


# 1) model

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_hat = torch.sigmoid(self.linear(x))
        return y_hat


model = LogisticRegression(n_features)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()  # Binary Cross Entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward n loss
    y_hat = model(X_train)

    loss = criterion(y_hat, y_train)

    # backward
    loss.backward()

    # update weights
    optimizer.step()

    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch: {epoch + 1}, loss: {loss.item():.4f}')

with torch.no_grad():
    y_hat = model(X_test)
    y_hat_class = y_hat.round()
    acc = y_hat_class.eq(y_test).sum() / float(y_test.shape[0]) *100
    print(f'accuracy = {acc:.2f}%')
    # model.eval()

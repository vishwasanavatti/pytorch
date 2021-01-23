import torch
import torch.nn as nn

x = torch.tensor([2.0, 1.0, 0.1])
output = torch.softmax(x, dim=0)
print(output)

loss = nn.CrossEntropyLoss()

# Y = torch.tensor([0])
Y = torch.tensor([2, 0, 1])
# n_samples * n_classes
# Y_hat_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_hat_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.2, 3.0, 0.1]])
# Y_hat_bad = torch.tensor([[0.5, 2.0, 0.4]])
Y_hat_bad = torch.tensor([[2.1, 1.0, 0.4], [0.1, 1.0, 2.1], [0.2, 3.0, 0.4]])

l1 = loss(Y_hat_good, Y)
l2 = loss(Y_hat_bad, Y)

print(l1.item(), l2.item())

_, predictions1 = torch.max(Y_hat_good, 1)
_, predictions2 = torch.max(Y_hat_bad, 1)
print(predictions1, predictions2)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
import sys
import torch.nn.functional as F

writer = SummaryWriter("runs/mnist2")
# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 784  # 28x28
hidden_size = 100
n_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.01

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
# print(samples.shape, labels.shape)

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(samples[i][0], cmap='gray')

img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnsit_images', img_grid)
writer.close()


# sys.exit()


# plt.show()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

        return out


model = NeuralNet(input_size, hidden_size, n_classes)

# loss n optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

writer.add_graph(model, samples.reshape(samples.shape[0], -1))
writer.close()
# sys.exit()

# training loop
n_total_steps = len(train_loader)

running_loss = 0.0
running_correct_pred = 0.0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(images.shape[0], -1).to(device)
        labels = labels.to(device)

        # forward
        output = model(images)
        loss = criterion(output, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predictions = torch.max(output, 1)
        running_correct_pred += (predictions == labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'epoch: {epoch + 1}/{num_epochs}, step: {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')
            writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
            writer.add_scalar('accuracy', running_correct_pred / 100, epoch * n_total_steps + i)
            running_loss = 0.0
            running_correct_pred = 0.0
            writer.close()

# testing
labels_list = []
preds = []
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for images, labels in test_loader:
        images = images.reshape(images.shape[0], -1).to(device)
        labels = labels.to(device)

        output = model(images)
        _, predictions = torch.max(output, 1)  # returns value n index
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

        class_preds = [F.softmax(out, dim=0) for out in output]

        labels_list.append(predictions)
        preds.append(class_preds)

    labels_list = torch.cat(labels_list)
    preds = torch.cat([torch.stack(batch) for batch in preds])

    acc = n_correct / n_samples * 100.0
    print(f'Accuracy = {acc}')

    classes = range(10)
    for i in classes:
        labels_i = labels_list == i
        preds_i = preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        writer.close()

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math


class WineDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])  # n_samples, 1
        self.n_samples = xy.shape[0]

    def __getitem__(self, item):
        # dataset[0]
        return self.x[item], self.y[item]

    def __len__(self):
        # len(dataset)
        return self.n_samples


dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

# training loop
num_epochs = 2
total_samples = len(dataset)
n_iters = math.ceil(total_samples / 4)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward n backward n update
        if (i + 1) % 5 == 0:
            print(f'epoch: {epoch + 1}/ {num_epochs}, step {i + 1}/{n_iters}, inputs {inputs.shape}')

# torchvision.datasets.MNIST()

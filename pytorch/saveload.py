import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_inp_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_inp_features, 1)

    def forward(self, x):
        y_hat = torch.sigmoid(self.linear(x))
        return y_hat


model = Model(n_inp_features=6)

# print(model.state_dict())
# FILE = "model1.pth"
# torch.save(model, FILE)
# torch.save(model.state_dict(), FILE)

# lazy option to access saved model
# model = torch.load(FILE)
# model.eval()
#
# for param in model.parameters():
#     print(param)

# preferred way
# loaded_model = Model(n_inp_features=6)
# loaded_model.load_state_dict(torch.load(FILE))
# loaded_model.eval()
#
# for param in model.parameters():
#      print(param)

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(optimizer.state_dict())

# svaing checkpoint
# checkpoint = {
#     'epoch': 90,
#     'model': model.state_dict(),
#     'optim_state': optimizer.state_dict()
# }

# torch.save(checkpoint, "checkpoint.pth")

loaded_checkpoint = torch.load('checkpoint.pth')
epoch = loaded_checkpoint['epoch']
model = Model(n_inp_features=6)
optimizer_loaded = torch.optim.SGD(model.parameters(), lr=0)

model.load_state_dict(loaded_checkpoint['model'])
optimizer.load_state_dict(loaded_checkpoint['optim_state'])

print(optimizer.state_dict())

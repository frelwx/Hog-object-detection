import torch
import torchvision
import torch.nn as nn
class linear_classfier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2496, 512)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(512, 2)
    def forward(self, x):
        # print(x.shape, x.device)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

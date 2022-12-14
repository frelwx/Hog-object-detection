import torch
import torchvision
import torch.nn as nn
class linear_classfier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2496, 2)
    def forward(self, x):
        # print(x.shape, x.device)
        x = self.linear(x)
        return x

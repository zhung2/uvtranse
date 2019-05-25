import torch
from torch.nn import Module

class SoftCrossEntropyLoss(Module):
    def __init__(self, size_average=True):
        super(SoftCrossEntropyLoss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        # Target is is smooth version of one hot encoding
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(-target * logsoftmax(input), dim=1)

        if self.size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)

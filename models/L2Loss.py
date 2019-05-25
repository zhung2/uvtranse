import torch
import torch.nn as nn

class L2Loss(nn.Module): 
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, output, target):
        #output_norm = torch.norm(output, p=2, dim=1)
        #lossvalue = torch.clamp(output_norm - target, min=0.0)
        #lossvalue = torch.clamp(output - target, min=0.0) ** 2
        lossvalue = torch.clamp(output ** 2 - target ** 2, min=0.0) 
        #lossvalue = (output - target) ** 2
        lossvalue = torch.mean(lossvalue)
        return lossvalue

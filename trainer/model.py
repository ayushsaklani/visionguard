import torch 
import torchvision
import torch.nn as nn
from torchvision.models import swin_v2_t, Swin_V2_T_Weights



def get_backbone(emb_size):
    backbone = swin_v2_t(Swin_V2_T_Weights.IMAGENET1K_V1)
    backbone.head = nn.Linear(768,128,bias=True)
    return backbone

class PersonAttributeHead(nn.Module):
    def __init__(self, in_channels, num_attributes=26):
        super(PersonAttributeHead, self).__init__()
        
        # Linear layer for attribute prediction
        self.conv = nn.Conv1d(in_channels,in_channels,3,1,1)
        self.batchNorm = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_channels, num_attributes)

    def forward(self, x):
        # x = self.relu(self.batchNorm(self.conv(x)))
        # x = self.relu(self.batchNorm(self.conv(x)))
        # Apply the linear layer
        output = self.linear(x)

        return output


class VisionGuard(nn.Module):
    def __init__(self,emb_size=128,num_attr=26):
        super(VisionGuard, self).__init__()
        
        self.backbone = get_backbone(emb_size)
        self.head_attr = PersonAttributeHead(emb_size,num_attr)
        
    def forward(self,x):
        x = self.backbone(x)
        
        out = self.head_attr(x)
        return out


import torch
from torch import nn

from element import ConvLayer


class TopDownLayer(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, 1, stride=1, padding=0)
        self.conv2 = ConvLayer(out_channels, out_channels*2, 3, stride=1, padding=1)
        self.conv3 = ConvLayer(out_channels*2, out_channels, 1, stride=1, padding=0)
        self.conv4 = ConvLayer(out_channels, out_channels*2, 3, stride=1, padding=1)
        self.conv5 = ConvLayer(out_channels*2, out_channels, 1, stride=1, padding=0)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out



class YOLOv3_FPN(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super().__init__()
        self.topdown1 = TopDownLayer(1024, 512)
        self.conv1 = ConvLayer(512, 256, 1, stride=1, padding=0)
        self.topdown2 = TopDownLayer(768, 256)
        self.conv2 = ConvLayer(256, 128, 1, stride=1, padding=0)
        self.topdown3 = TopDownLayer(384, 128)
        self.upsample = nn.Upsample(scale_factor=2)


    def forward(self, x1, x2, x3):
        C1 = self.topdown1(x1)
        P1 = self.upsample(self.conv1(C1))
        C2 = self.topdown2(torch.cat((P1, x2), dim=1))
        P2 = self.upsample(self.conv2(C2))
        C3 = self.topdown3(torch.cat((P2, x3), dim=1))
        return C1, C2, C3

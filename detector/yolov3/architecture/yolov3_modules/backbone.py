from torch import nn

from element import ConvLayer, ResBlock



class Darknet53(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayer(3, 32, 3, stride=1, padding=1)
        self.res_block1 = self._build_Conv_and_ResBlock(32, 64, 1)
        self.res_block2 = self._build_Conv_and_ResBlock(64, 128, 2)
        self.res_block3 = self._build_Conv_and_ResBlock(128, 256, 8)
        self.res_block4 = self._build_Conv_and_ResBlock(256, 512, 8)
        self.res_block5 = self._build_Conv_and_ResBlock(512, 1024, 4)


    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.res_block1(tmp)
        tmp = self.res_block2(tmp)
        out3 = self.res_block3(tmp)
        out2 = self.res_block4(out3)
        out1 = self.res_block5(out2)
        return out1, out2, out3


    def _build_Conv_and_ResBlock(self, in_channels, out_channels, num_block):
        model = nn.Sequential()
        model.add_module("conv", ConvLayer(in_channels, out_channels, 3, stride=2, padding=1))
        for idx in range(num_block):
            model.add_module(f"res{idx}", ResBlock(out_channels))
        return model

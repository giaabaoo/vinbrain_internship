import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet50_Weights

resnet = torchvision.models.resnet.resnet50(ResNet50_Weights.DEFAULT)


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x

class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=1):
        super().__init__()
        resnet = torchvision.models.resnet.resnet50(ResNet50_Weights.DEFAULT)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

class ConvBlock(nn.Module):
    def __init__(self, chan_in, 
                       chan_out, 
                       padding = 1, 
                       kernel_size = 3, 
                       stride = 1
                ):
        super().__init__()
        self.conv = nn.Conv2d(chan_in, 
                              chan_out, 
                              padding = padding, 
                              kernel_size = kernel_size, 
                              stride = stride)
        self.bn = nn.BatchNorm2d(chan_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class Bridge(nn.Module):
    def __init__(self, chan_in, chan_out):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(chan_in, chan_out),
            ConvBlock(chan_out, chan_out)
        )

    def forward(self, x):
        return self.bridge(x)
    
class UpBlock(nn.Module):    
    def __init__(self, chan_in, 
                       chan_out, 
                       upconv_in = None, 
                       upconv_out = None,
                ):
        super().__init__()
        if upconv_in == None:
            upconv_in = chan_in
        if upconv_out == None:
            upconv_out = chan_out
        self.upsample = nn.ConvTranspose2d(upconv_in, upconv_out, kernel_size = 2, stride = 2)
        self.conv = nn.Sequential(ConvBlock(chan_in, chan_out),
                                  ConvBlock(chan_out, chan_out))

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x1 = torch.cat([x1, x2], 1)
        x1 = self.conv(x1)
        return x1
    
class UNetWithResNext101Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnext101 = torchvision.models.resnext101_32x8d(pretrained=True)
        # conv -> batchnorm -> reLU
        self.module0 = nn.Sequential(*list(resnext101.children()))[:3] 
        # max pool layer in resnet
        self.module0pool = list(resnext101.children())[3] 
        # collect bottleneck in resnet to use as block module in Unet
        encoder = []
        # there are 4 module nn.Sequential in Resnet50
        for module in list(resnext101.children()):
            if isinstance(module, nn.Sequential):
                encoder.append(module)
        self.encoder = nn.ModuleList(encoder)
        self.bridge = Bridge(2048, 2048)

        self.decoder = nn.ModuleList([
            UpBlock(2048, 1024),
            UpBlock(1024, 512),
            UpBlock(512, 256),
            UpBlock(192, 128, 256, 128),
            UpBlock(67, 64, 128, 64)
        ])
        self.outLayer = nn.Conv2d(64, 1, kernel_size = 1, stride = 1)

    def forward(self, x):
        tempStorage = dict()
        tempStorage["t_0"] = x
        # x.size(): 3x512x512
        x = self.module0(x)
        # x.size(): 64x256x256
        tempStorage["t_1"] = x
        x = self.module0pool(x)
        # x.size():64x128x128
        for idx, module in enumerate(self.encoder, start = 2):
            x = module(x)
            # 2, 3, 4, 5
            if idx == 5: 
                continue
            tempStorage[f"t_{idx}"] = x

        x = self.bridge(x)

        for idx, module in enumerate(self.decoder, start = 1):
            match_indice = 5 - idx
            temp_key = f"t_{match_indice}"
            x = module(x, tempStorage[temp_key])
        x = self.outLayer(x)
        return x

if __name__ == "__main__":
    model = UNetWithResnet50Encoder().cuda()
    inp = torch.rand((2, 3, 512, 512)).cuda()
    out = model(inp)
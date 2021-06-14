import torch
import torch.nn as nn
from pfrl.policies import SoftmaxCategoricalHead


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(inplace=True)
    )


class CategoricalHead(nn.Module):
    def forward(self, x):
        return torch.distributions.Categorical(probs=x.softmax(dim=1))


class A2CNet(nn.Module):

    def __init__(self, n_filters=8):
        super().__init__()

        self.dconv_down1 = double_conv(3, n_filters)
        self.dconv_down2 = double_conv(n_filters, n_filters * 2)
        self.dconv_down3 = double_conv(n_filters * 2, n_filters * 4)
        self.dconv_down4 = double_conv(n_filters * 4, n_filters * 8)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(n_filters * 4 + n_filters * 8, n_filters * 4)
        self.dconv_up2 = double_conv(n_filters * 2 + n_filters * 4, n_filters * 2)
        self.dconv_up1 = double_conv(n_filters * 2 + n_filters, n_filters)

        self.conv_last = nn.Conv2d(n_filters, 1, 1)
        self.cat_softmax = CategoricalHead(),

        self.v_conv1 = nn.Sequential(
            nn.Conv2d(n_filters * 8, n_filters * 2, 3),
            nn.ReLU(inplace=True)
        )
        self.v_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.v_linear1 = nn.Linear(n_filters * 2, n_filters)
        self.v_linear2 = nn.Linear(n_filters, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        y = self.v_conv1(x)
        y = self.v_max_pool(y)
        y = self.v_linear1(y.reshape(y.size(0), -1))
        val = self.v_linear2(y)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        act = self.cat_softmax[0](x.abs().reshape(x.size(0), -1))

        return act, val

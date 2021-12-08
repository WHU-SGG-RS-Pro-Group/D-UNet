import torch
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.init

#
class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes//4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes//4, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, in_planes):
        super(SpatialAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_planes, in_planes//2, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 2, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(x)))
        return self.sigmoid(avg_out)


class mrcam(nn.Module):
    def __init__(self, hs_channels,mschannels, bias=True):
        super(mrcam, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(hs_channels+mschannels, 64, kernel_size=3, stride=1, padding=1, bias=bias),

                     nn.LeakyReLU(0.2)
         )

        self.conv0_1 = nn.Parameter(torch.FloatTensor([1]),requires_grad=True)

        self.conv1 = nn.Sequential(

                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias),

        )

        self.conv1_1 = nn.Parameter(torch.FloatTensor([1]),requires_grad=True)

        self.conv2 = nn.Sequential(

                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias),

                nn.LeakyReLU(0.2),

                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias),

        )
        self.conv1_2 = nn.Parameter(torch.FloatTensor([1]),requires_grad=True)

        self.conv3=nn.Sequential(

                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias),

                nn.LeakyReLU(0.2),

                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias),

                nn.LeakyReLU(0.2),

                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias),

        )
        self.conv1_3 = nn.Parameter(torch.FloatTensor([1]),requires_grad=True)

        self.cb1 = nn.Conv2d(64*2, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.cb2 = nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1, bias=bias)

        self.cb = nn.Sequential(
            nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(0.2)
        )

        self.channel_3_3 = ChannelAttention(64)
        self.spatial_3_3 = SpatialAttention(64)
        self.factor_3 = nn.Parameter(torch.FloatTensor([1]),requires_grad=True)
        self.channel_5_5 = ChannelAttention(64)
        self.spatial_5_5 = SpatialAttention(64)
        self.factor_5 = nn.Parameter(torch.FloatTensor([1]),requires_grad=True)
        self.channel_7_7 = ChannelAttention(64)
        self.spatial_7_7 = SpatialAttention(64)
        self.factor_7 = nn.Parameter(torch.FloatTensor([1]),requires_grad=True)
        self.fusion = nn.Sequential(

                nn.Conv2d(64*3, 64, kernel_size=1, stride=1, padding=0, bias=bias),

                nn.LeakyReLU(0.2),

                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias),

        )

    def forward(self, x):
        x0 = x = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        cb1 = torch.cat((x1*self.conv1_1, x2*self.conv1_2), dim=1)
        cb2 = torch.cat((x1*self.conv1_1, x2*self.conv1_2, x3*self.conv1_3), dim=1)

        cb1 = self.cb1(cb1)
        cb2 = self.cb2(cb2)
        cb3 = torch.cat((x1*self.conv1_1,cb1,cb2),dim=1)
        x = self.cb(cb3)

        atten1 = self.channel_3_3(x)*self.spatial_3_3(x)*x + x
        atten2 = self.channel_5_5(x)*self.spatial_5_5(x)*x + x
        atten3 = self.channel_7_7(x)*self.spatial_7_7(x)*x + x
        atten1 = self.factor_3*atten1
        atten2 = self.factor_5*atten2
        atten3 = self.factor_7*atten3

        atten = torch.cat((atten1, atten2, atten3), dim=1)
        x = self.fusion(atten) + x0
        return x


class UNet(nn.Module):
    def __init__(self, hschannels, mschannels, bilinear=True):
        super(UNet, self).__init__()
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        #hsi 主网络
        self.inc = DoubleConv(64, 64)
        self.mscsam = mrcam(hschannels, mschannels)
        self.down1 = Down(64+32, 128)
        self.down2 = Down(128+64, 256)
        self.down3 = Down(256+128, 512 // factor)
        # self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(256+(512 // factor)+256// factor, 256 // factor, 256, bilinear)
        self.up2 = Up(128+256 // factor+128// factor, 128 // factor, 128, bilinear)
        self.up3 = Up(64+128 // factor+64// factor, 64, 64, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, hschannels)
        #ms特征提取网络
        self.incm = DoubleConv(mschannels, 32)
        self.downm1 = Down(32, 64)
        self.downm2 = Down(64, 128)
        self.downm3 = Down(128, 256 // factor)
        # self.down4 = Down(512, 1024 // factor)
        # self.upm1 = Up(256, 128 // factor, bilinear)
        # self.upm2 = Up(128, 64 // factor, bilinear)
        # self.upm3 = Up(64, 32, bilinear)
        self.upm1 = Upm(256 // factor, 128 // factor, bilinear)
        self.upm2 = Upm(128 // factor, 64 // factor, bilinear)
        self.upm3 = Upm(64 // factor, 64, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        self.outmc = OutConv(64, hschannels)

    def forward(self, x,y):
        y0=y
        y1 = self.incm(y)
        y2 = self.downm1(y1)
        y3 = self.downm2(y2)
        y4 = self.downm3(y3)
        # y5 = self.upm1(y4, y3)
        # y6 = self.upm2(y5, y2)
        # y = self.upm3(y6, y1)
        y5 = self.upm1(y4)
        y6 = self.upm2(y5)
        y = self.upm3(y6)
        logitsy = self.outmc(y)

        x0 = x
        # 嵌入多尺度注意力
        # x1 = self.mscsam(x)

        x1 = self.mscsam(torch.cat((x,y0),dim=1))
        x1 = self.inc(x1)

        x2 = self.down1(torch.cat((x1, y1), dim=1))
        x3 = self.down2(torch.cat((x2, y2), dim=1))
        x4 = self.down3(torch.cat((x3, y3), dim=1))
        # x5 = self.down4(x4)
        x = self.up1(torch.cat((x4, y4),dim=1),x3)
        x = self.up2(torch.cat((x, y5),dim=1),x2)
        x = self.up3(torch.cat((x, y6),dim=1),x1)
        # x = self.up4(x, x1)
        logits = self.outc(x)
        # return logits+x0, logits+x0

        return logits+x0, logits+x0+logitsy


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Upsample(mode='bilinear',scale_factor=1/2),
            DoubleConv(in_channels, out_channels)
            # nn.MaxPool2d(2),
            # DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels-channels, (in_channels-channels) // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv((in_channels-channels) // 2+channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class Upm(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels)


    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),

        )
    def forward(self, x):
        return self.conv(x)


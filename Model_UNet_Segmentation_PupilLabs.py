import torch
import torch.nn as nn
import parameters_PupilLabs

first_8 = 8
first_16 = 16
first_32 = 32
Size_X = parameters_PupilLabs.Size_X
Size_Y = parameters_PupilLabs.Size_Y

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet4f16ch(nn.Module):

    def __init__(self):
        super().__init__()
        self.dconv_down1 = double_conv(1, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 64)
        self.dconv_down4 = double_conv(64, 128)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.upsample3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)

        self.dconv_up3 = double_conv(64 + 64, 64)
        self.dconv_up2 = double_conv(32 + 32, 32)
        self.dconv_up1 = double_conv(16 + 16, 16)

        self.conv_last = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1)
        self.soft_max = nn.Softmax(dim=1)
        # self.conv2d = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        d1 = self.maxpool1(conv1)

        conv2 = self.dconv_down2(d1)
        d2 = self.maxpool2(conv2)

        conv3 = self.dconv_down3(d2)
        d3 = self.maxpool3(conv3)

        conv4 = self.dconv_down4(d3)

        u3 = self.upsample3(conv4)
        cat3 = torch.cat([u3, conv3], dim=1)

        conv_3 = self.dconv_up3(cat3)
        u2 = self.upsample2(conv_3)
        cat2 = torch.cat([u2, conv2], dim=1)

        conv_2 = self.dconv_up2(cat2)
        u1 = self.upsample1(conv_2)
        cat1 = torch.cat([u1, conv1], dim=1)

        conv_1 = self.dconv_up1(cat1)
        ch_2 = self.conv_last(conv_1)
        out = self.soft_max(ch_2)

        return out



class UNet4f32ch(nn.Module):

    def __init__(self):
        super().__init__()
        self.dconv_down1 = double_conv(1, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)
        self.dconv_down4 = double_conv(128, 256)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)

        self.dconv_up3 = double_conv(128 + 128, 128)
        self.dconv_up2 = double_conv(64 + 64, 64)
        self.dconv_up1 = double_conv(32 + 32, 32)

        self.conv_last = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
        self.soft_max = nn.Softmax(dim=1)
        # self.conv2d = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        d1 = self.maxpool1(conv1)

        conv2 = self.dconv_down2(d1)
        d2 = self.maxpool2(conv2)

        conv3 = self.dconv_down3(d2)
        d3 = self.maxpool3(conv3)

        conv4 = self.dconv_down4(d3)

        u3 = self.upsample3(conv4)
        cat3 = torch.cat([u3, conv3], dim=1)

        conv_3 = self.dconv_up3(cat3)
        u2 = self.upsample2(conv_3)
        cat2 = torch.cat([u2, conv2], dim=1)

        conv_2 = self.dconv_up2(cat2)
        u1 = self.upsample1(conv_2)
        cat1 = torch.cat([u1, conv1], dim=1)

        conv_1 = self.dconv_up1(cat1)
        # ch_2 = self.conv_last(conv_1)
        out = self.conv_last(conv_1)
        # out = self.soft_max(ch_2)

        return out

class UNet4f16ch_softmax(nn.Module):

    def __init__(self):
        super().__init__()
        self.first = 16

        self.dconv_down1 = double_conv(1, self.first)
        self.dconv_down2 = double_conv(self.first, self.first*2)
        self.dconv_down3 = double_conv(self.first*2, self.first*4)
        self.dconv_down4 = double_conv(self.first*4, self.first*8)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.upsample3 = nn.ConvTranspose2d(in_channels=self.first*8, out_channels=self.first*4, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(in_channels=self.first*4, out_channels=self.first*2, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(in_channels=self.first*2, out_channels=self.first, kernel_size=2, stride=2)

        self.dconv_up3 = double_conv(self.first*4 + self.first*4, self.first*4)
        self.dconv_up2 = double_conv(self.first*2 + self.first*2, self.first*2)
        self.dconv_up1 = double_conv(self.first + self.first, self.first)

        self.conv_last = nn.Conv2d(in_channels=self.first, out_channels=2, kernel_size=3, padding=1)
        self.soft_max = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()
        # self.conv2d = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        d1 = self.maxpool1(conv1)

        conv2 = self.dconv_down2(d1)
        d2 = self.maxpool2(conv2)

        conv3 = self.dconv_down3(d2)
        d3 = self.maxpool3(conv3)

        conv4 = self.dconv_down4(d3)

        u3 = self.upsample3(conv4)
        cat3 = torch.cat([u3, conv3], dim=1)

        conv_3 = self.dconv_up3(cat3)
        u2 = self.upsample2(conv_3)
        cat2 = torch.cat([u2, conv2], dim=1)

        conv_2 = self.dconv_up2(cat2)
        u1 = self.upsample1(conv_2)
        cat1 = torch.cat([u1, conv1], dim=1)

        conv_1 = self.dconv_up1(cat1)
        ch_2 = self.conv_last(conv_1)
        out = self.soft_max(ch_2)

        return out


#
class UNet4f8ch_sigmoid(nn.Module):

    def __init__(self):
        super().__init__()
        self.first = 8
        self.dconv_down1 = double_conv(1, self.first)
        self.dconv_down2 = double_conv(self.first, self.first*2)
        self.dconv_down3 = double_conv(self.first*2, self.first*4)
        self.dconv_down4 = double_conv(self.first*4, self.first*8)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.upsample3 = nn.ConvTranspose2d(in_channels=self.first*8, out_channels=self.first*4, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(in_channels=self.first*4, out_channels=self.first*2, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(in_channels=self.first*2, out_channels=self.first, kernel_size=2, stride=2)

        self.dconv_up3 = double_conv(self.first*4 + self.first*4, self.first*4)
        self.dconv_up2 = double_conv(self.first*2 + self.first*2, self.first*2)
        self.dconv_up1 = double_conv(self.first + self.first, self.first)

        self.conv_last = nn.Conv2d(in_channels=self.first, out_channels=1, kernel_size=3, padding=1)
        # self.soft_max = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        # self.conv2d = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        d1 = self.maxpool1(conv1)

        conv2 = self.dconv_down2(d1)
        d2 = self.maxpool2(conv2)

        conv3 = self.dconv_down3(d2)
        d3 = self.maxpool3(conv3)

        conv4 = self.dconv_down4(d3)

        u3 = self.upsample3(conv4)
        cat3 = torch.cat([u3, conv3], dim=1)

        conv_3 = self.dconv_up3(cat3)
        u2 = self.upsample2(conv_3)
        cat2 = torch.cat([u2, conv2], dim=1)

        conv_2 = self.dconv_up2(cat2)
        u1 = self.upsample1(conv_2)
        cat1 = torch.cat([u1, conv1], dim=1)

        conv_1 = self.dconv_up1(cat1)
        ch_2 = self.conv_last(conv_1)
        out = self.sigmoid(ch_2)

        return out

class UNet3f16ch_sigmoid(nn.Module):

    def __init__(self):
        super().__init__()
        self.first = 16
        self.dconv_down1 = double_conv(1, self.first)
        self.dconv_down2 = double_conv(self.first, self.first*2)
        self.dconv_down3 = double_conv(self.first*2, self.first*4)
        self.dconv_down4 = double_conv(self.first*4, self.first*8)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.upsample3 = nn.ConvTranspose2d(in_channels=self.first*8, out_channels=self.first*4, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(in_channels=self.first*4, out_channels=self.first*2, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(in_channels=self.first*2, out_channels=self.first, kernel_size=2, stride=2)

        self.dconv_up3 = double_conv(self.first*4 + self.first*4, self.first*4)
        self.dconv_up2 = double_conv(self.first*2 + self.first*2, self.first*2)
        self.dconv_up1 = double_conv(self.first + self.first, self.first)

        self.conv_last = nn.Conv2d(in_channels=self.first, out_channels=1, kernel_size=3, padding=1)
        # self.soft_max = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        # self.conv2d = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        d1 = self.maxpool1(conv1)

        conv2 = self.dconv_down2(d1)
        d2 = self.maxpool2(conv2)

        conv3 = self.dconv_down3(d2)
        d3 = self.maxpool3(conv3)

        conv4 = self.dconv_down4(d3)

        u3 = self.upsample3(conv4)
        cat3 = torch.cat([u3, conv3], dim=1)

        conv_3 = self.dconv_up3(cat3)
        u2 = self.upsample2(conv_3)
        cat2 = torch.cat([u2, conv2], dim=1)

        conv_2 = self.dconv_up2(cat2)
        u1 = self.upsample1(conv_2)
        cat1 = torch.cat([u1, conv1], dim=1)

        conv_1 = self.dconv_up1(cat1)
        ch_2 = self.conv_last(conv_1)
        out = self.sigmoid(ch_2)

        return out

class UNet4f16ch_sigmoid(nn.Module):

    def __init__(self):
        super().__init__()
        self.first = 16

        self.dconv_down1 = double_conv(1, self.first)
        self.dconv_down2 = double_conv(self.first, self.first*2)
        self.dconv_down3 = double_conv(self.first*2, self.first*4)
        self.dconv_down4 = double_conv(self.first*4, self.first*8)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.upsample3 = nn.ConvTranspose2d(in_channels=self.first*8, out_channels=self.first*4, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(in_channels=self.first*4, out_channels=self.first*2, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(in_channels=self.first*2, out_channels=self.first, kernel_size=2, stride=2)

        self.dconv_up3 = double_conv(self.first*4 + self.first*4, self.first*4)
        self.dconv_up2 = double_conv(self.first*2 + self.first*2, self.first*2)
        self.dconv_up1 = double_conv(self.first + self.first, self.first)

        self.conv_last = nn.Conv2d(in_channels=self.first, out_channels=1, kernel_size=3, padding=1)
        # self.soft_max = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        # self.conv2d = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        d1 = self.maxpool1(conv1)

        conv2 = self.dconv_down2(d1)
        d2 = self.maxpool2(conv2)

        conv3 = self.dconv_down3(d2)
        d3 = self.maxpool3(conv3)

        conv4 = self.dconv_down4(d3)

        u3 = self.upsample3(conv4)
        cat3 = torch.cat([u3, conv3], dim=1)

        conv_3 = self.dconv_up3(cat3)
        u2 = self.upsample2(conv_3)
        cat2 = torch.cat([u2, conv2], dim=1)

        conv_2 = self.dconv_up2(cat2)
        u1 = self.upsample1(conv_2)
        cat1 = torch.cat([u1, conv1], dim=1)

        conv_1 = self.dconv_up1(cat1)
        ch_2 = self.conv_last(conv_1)
        out = self.sigmoid(ch_2)

        return out

class UNet4f32ch_sigmoid(nn.Module):

    def __init__(self):
        super().__init__()
        self.first = 32

        self.dconv_down1 = double_conv(1, self.first)
        self.dconv_down2 = double_conv(self.first, self.first*2)
        self.dconv_down3 = double_conv(self.first*2, self.first*4)
        self.dconv_down4 = double_conv(self.first*4, self.first*8)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.upsample3 = nn.ConvTranspose2d(in_channels=self.first*8, out_channels=self.first*4, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(in_channels=self.first*4, out_channels=self.first*2, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(in_channels=self.first*2, out_channels=self.first, kernel_size=2, stride=2)

        self.dconv_up3 = double_conv(self.first*4 + self.first*4, self.first*4)
        self.dconv_up2 = double_conv(self.first*2 + self.first*2, self.first*2)
        self.dconv_up1 = double_conv(self.first + self.first, self.first)

        self.conv_last = nn.Conv2d(in_channels=self.first, out_channels=1, kernel_size=3, padding=1)
        # self.soft_max = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        # self.conv2d = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        d1 = self.maxpool1(conv1)

        conv2 = self.dconv_down2(d1)
        d2 = self.maxpool2(conv2)

        conv3 = self.dconv_down3(d2)
        d3 = self.maxpool3(conv3)

        conv4 = self.dconv_down4(d3)

        u3 = self.upsample3(conv4)
        cat3 = torch.cat([u3, conv3], dim=1)

        conv_3 = self.dconv_up3(cat3)
        u2 = self.upsample2(conv_3)
        cat2 = torch.cat([u2, conv2], dim=1)

        conv_2 = self.dconv_up2(cat2)
        u1 = self.upsample1(conv_2)
        cat1 = torch.cat([u1, conv1], dim=1)

        conv_1 = self.dconv_up1(cat1)
        ch_2 = self.conv_last(conv_1)
        out = self.sigmoid(ch_2)

        return out



class UNet4f16ch_seg_reg(nn.Module):

    def __init__(self):
        super().__init__()
        self.first = 16

        self.dconv_down1 = double_conv(1, self.first)
        self.dconv_down2 = double_conv(self.first, self.first*2)
        self.dconv_down3 = double_conv(self.first*2, self.first*4)
        self.dconv_down4 = double_conv(self.first*4, self.first*8)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.upsample3 = nn.ConvTranspose2d(in_channels=self.first*8, out_channels=self.first*4, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(in_channels=self.first*4, out_channels=self.first*2, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(in_channels=self.first*2, out_channels=self.first, kernel_size=2, stride=2)

        self.dconv_up3 = double_conv(self.first*4 + self.first*4, self.first*4)
        self.dconv_up2 = double_conv(self.first*2 + self.first*2, self.first*2)
        self.dconv_up1 = double_conv(self.first + self.first, self.first)

        self.conv_last = nn.Conv2d(in_channels=self.first, out_channels=1, kernel_size=3, padding=1)

        self.Reg_conv_1 = double_conv(self.first, self.first*2)
        self.Reg_maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.Reg_conv_2 = double_conv(self.first*2, self.first*4)
        self.Reg_maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.Reg_conv_3 = double_conv(self.first*4, self.first*8)
        self.Reg_maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.Fc1 = nn.Linear(self.first*8*int(Size_X/8)*int(Size_Y/8), 256)
        self.Fc2 = nn.Linear(256, 512)
        self.reg_out = nn.Linear(512, 2)

        # self.soft_max = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        # self.conv2d = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        d1 = self.maxpool1(conv1)

        conv2 = self.dconv_down2(d1)
        d2 = self.maxpool2(conv2)

        conv3 = self.dconv_down3(d2)
        d3 = self.maxpool3(conv3)

        conv4 = self.dconv_down4(d3)

        u3 = self.upsample3(conv4)
        cat3 = torch.cat([u3, conv3], dim=1)

        conv_3 = self.dconv_up3(cat3)
        u2 = self.upsample2(conv_3)
        cat2 = torch.cat([u2, conv2], dim=1)

        conv_2 = self.dconv_up2(cat2)
        u1 = self.upsample1(conv_2)
        cat1 = torch.cat([u1, conv1], dim=1)

        conv_1 = self.dconv_up1(cat1)
        ch_2= self.conv_last(conv_1)
        out_seg = self.sigmoid(ch_2)

        reg_conv_1 = self.Reg_conv_1(conv1)
        reg_max_1 = self.Reg_maxpool1(reg_conv_1)
        reg_conv_2 = self.Reg_conv_2(reg_max_1)
        reg_max_2 = self.Reg_maxpool2(reg_conv_2)
        reg_conv_3 = self.Reg_conv_3(reg_max_2)
        reg_max_3 = self.Reg_maxpool3(reg_conv_3)
        reg_max_3 = reg_max_3.view(-1, self.first*8*int(Size_X/8)*int(Size_Y/8))
        fc1 = self.Fc1(reg_max_3)
        fc2 = self.Fc2(fc1)
        out_reg = self.reg_out(fc2)

        return out_seg, out_reg, ch_2
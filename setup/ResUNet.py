import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.double_conv(x)
        out = self.relu(out + identity)
        return self.down_sample(out), out

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class ResUNet(nn.Module):
    def __init__(self, filters=[16,32,64,128,256], input_channels=3, output_channels=3):
        super(ResUNet, self).__init__()
        if len(filters) != 5:
            raise Exception("Filter list size {s}, expected 5!".format(len(filters)))
        self.down_conv1 = ResBlock(input_channels, filters[0])
        self.down_conv2 = ResBlock(filters[0], filters[1])
        self.down_conv3 = ResBlock(filters[1], filters[2])
        self.down_conv4 = ResBlock(filters[2], filters[3])

        self.double_conv = DoubleConv(filters[3], filters[4])

        self.up_conv4 = UpBlock(filters[3] + filters[4], filters[3])
        self.up_conv3 = UpBlock(filters[2] + filters[3], filters[2])
        self.up_conv2 = UpBlock(filters[1] + filters[2], filters[1])
        self.up_conv1 = UpBlock(filters[0] + filters[1], filters[0])

        self.conv_last = nn.Conv2d(filters[0], output_channels, kernel_size=1)

    def forward(self, inputs):
        x, skip1_out = self.down_conv1(inputs)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)

        # we need to use raw output for cross_entropy_loss (already have logsoftmax)
        # x = F.softmax(x, dim=1)
        # x = torch.argmax(x, dim=1)

        return x

def test():
    x = torch.rand((2, 3, 512, 512))
    model = ResUNet()
    preds = model(x)
    print(x.shape)
    print(preds.shape)
    print(preds)

if __name__ == "__main__":
    test()

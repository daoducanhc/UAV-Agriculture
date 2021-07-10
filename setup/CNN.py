import torch
import torch.nn as nn
from math import sqrt

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=1, padding=1, bias=True)

class CNN(nn.Module):
    def __init__(self, input_channel=3, output_channel=3):
        super(CNN, self).__init__()

        self.out_channels = output_channel
        self.conv1 = conv3x3(input_channel, out_channels=64)
        self.conv = conv3x3(64, 64)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.output_conv = conv3x3(in_channels=64,  out_channels=self.out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv(out)
        out = self.conv1_bn(out)
        out = self.relu(out)
        out = self.output_conv(out)
        return out

def test():
    x = torch.rand((1, 3, 512, 512))
    model = CNN()
    preds = model(x)
    print(x.shape)
    print(preds.shape)
    print(preds)

if __name__ == "__main__":
    test()

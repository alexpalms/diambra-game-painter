from torch import nn

# Image Transform Network
class ImageTransformNet(nn.Module):
    def __init__(self):
        super(ImageTransformNet, self).__init__()

        # Non-linearities
        self.relu = nn.ReLU6(inplace=True)

        # Initial convolution layers
        self.conv1 = ConvLayer(3, 16, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(16, affine=True)
        self.conv2 = ConvLayer(16, 16, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(16, affine=True)
        self.conv3 = ConvLayer(16, 16, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(16, affine=True)

        # Residual layers
        self.res = ResidualBlock(16)

        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(16, 16, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(16, affine=True)

        self.deconv2 = UpsampleConvLayer(16, 16, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(16, affine=True)

        self.deconv3 = ConvLayer(16, 3, kernel_size=3, stride=1)

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))

        y = self.res(y)

        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)

        return y

# Conv Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                padding=kernel_size//2)

    def forward(self, x):
        out = self.conv2d(x)
        return out


# Upsample Conv Layer
class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                padding=kernel_size//2)

    def forward(self, x):
        if self.upsample:
            x = nn.functional.interpolate(x, mode='nearest', scale_factor=self.upsample)
        out = self.conv2d(x)
        return out

class ResidualBlock(nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU6(inplace=True)

        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))

        out = self.in2(self.conv2(out))

        out = out + residual
        out = self.relu(out)
        return out

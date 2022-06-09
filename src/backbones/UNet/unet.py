import torch
from torch import nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2,2))

    def forward(self, x):
        x = self.conv1(x)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, enc_channels=(3, 64, 128, 256, 512), bottleneck=1024, num_classes=2):
        super().__init__()
        self.enc_channels = enc_channels
        self.dec_channels = [bottleneck] + list(reversed(enc_channels))[:-1]
        self.num_cls = num_classes
        self.depth = len(enc_channels) - 1

        self.encoders = nn.ModuleList(
            [
                encoder_block(self.enc_channels[i], self.enc_channels[i + 1]) for i in range(self.depth)
            ]
        )

        self.bottleneck = conv_block(self.enc_channels[-1], bottleneck)

        self.decoders = nn.ModuleList(
            [
                decoder_block(self.dec_channels[i], self.dec_channels[i + 1]) for i in range(self.depth)
            ]
        )

        self.out = nn.Conv2d(self.dec_channels[-1], self.num_cls, kernel_size=1, padding=0)


        # """ Encoder """
        # self.e1 = encoder_block(3, 64)
        # self.e2 = encoder_block(64, 128)
        # self.e3 = encoder_block(128, 256)
        # self.e4 = encoder_block(256, 512)

        # """ Bottleneck """
        # self.b = conv_block(512, 1024)

        # """ Decoder """
        # self.d1 = decoder_block(1024, 512)
        # self.d2 = decoder_block(512, 256)
        # self.d3 = decoder_block(256, 128)
        # self.d4 = decoder_block(128, 64)

        # """ Output """
        # self.out = nn.Conv2d(64, 19, kernel_size=1, padding=0)

    def forward(self, x):

        skips = []
        for i in range(self.depth):
            skip, x = self.encoders[i](x)
            skips.append(skip)
        
        x = self.bottleneck(x)

        for i in range(self.depth):
            x = self.decoders[i](x, skips[-i-1])

        out = self.out(x)


        # """ Encoder """
        # s1, p1 = self.e1(x)
        # s2, p2 = self.e2(p1)
        # s3, p3 = self.e3(p2)
        # s4, p4 = self.e4(p3)

        # """ Bottleneck """
        # b = self.b(p4)

        # """ Decoder """
        # d1 = self.d1(b, s4)
        # d2 = self.d2(d1, s3)
        # d3 = self.d3(d2, s2)
        # d4 = self.d4(d3, s1)

        # """ Output """
        # out = self.out(d4)

        return out
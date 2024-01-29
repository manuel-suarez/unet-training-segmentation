# Own scratch implementation
import torch
import torch.nn as nn

class Block1(nn.Module):
    def __init__(self):
        super().__init__()
        # First encoder block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        return x

class Block2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        return x

class Block3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        return x

class Block4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        return x

class Block5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        return x

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = Block1()
        self.block2 = Block2()
        self.block3 = Block3()
        self.block4 = Block4()
        self.block5 = Block5()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        r1 = self.block1(x)
        x = self.maxpool(r1)
        r2 = self.block2(x)
        x = self.maxpool(r2)
        r3 = self.block3(x)
        x = self.maxpool(r3)
        r4 = self.block4(x)
        x = self.maxpool(r4)
        x = self.block5(x)

        return x, [r1,r2,r3,r4]


def crop(x, shape=None):
    '''
    Function for cropping an image tensor: Given an image tensor and the new shape,
    crops to the center pixels (assumes that the input's size and the new size are
    even numbers).
    Parameters:
        image: image tensor of shape (batch size, channels, height, width)
        new_shape: a torch.Size object with the shape you want x to have
    '''
    _, _, h, w = x.shape
    _, _, h_new, w_new = shape

    ch, cw = h//2, w//2
    ch_new, cw_new = h_new//2, w_new//2
    x1 = int(cw - cw_new)
    y1 = int(ch - ch_new)
    x2 = int(x1 + w_new)
    y2 = int(y1 + h_new)
    return x[:, :, y1:y2, x1:x2]

class UpBlock4(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsampling = nn.Upsample(size=(57,57))
        self.upconv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=2)
        # concatenation
        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        self.relu2 = nn.ReLU()


    def forward(self, x, r):
        x = self.upsampling(x)
        x = self.upconv1(x)
        r = crop(r, x.shape)
        x = torch.cat([x, r], dim=1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        return x

class UpBlock3(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsampling = nn.Upsample(size=(105,105))
        self.upconv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=2)
        # concatenation
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.relu2 = nn.ReLU()

    def forward(self, x, r):
        x = self.upsampling(x)
        x = self.upconv1(x)
        r = crop(r, x.shape)
        x = torch.cat([x, r], dim=1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        return x

class UpBlock2(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsampling = nn.Upsample(size=(201,201))
        self.upconv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=2)
        # concatenation
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.relu2 = nn.ReLU()

    def forward(self, x, r):
        x = self.upsampling(x)
        x = self.upconv1(x)
        r = crop(r, x.shape)
        x = torch.cat([x, r], dim=1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        return x

class UpBlock1(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsampling = nn.Upsample(size=(393,393))
        self.upconv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2)
        # concatenation
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

    def forward(self, x, r):
        x = self.upsampling(x)
        x = self.upconv1(x)
        r = crop(r, x.shape)
        x = torch.cat([x, r], dim=1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block4 = UpBlock4()
        self.block3 = UpBlock3()
        self.block2 = UpBlock2()
        self.block1 = UpBlock1()

    def forward(self, x, r):
        x = self.block4(x, r[3])
        x = self.block3(x, r[2])
        x = self.block2(x, r[1])
        x = self.block1(x, r[0])

        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x, r = self.encoder(x)
        x = self.decoder(x, r)

        return x

if __name__ == '__main__':
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    model = Block1()
    input = torch.randn((1, 1, 572, 572))
    output1 = model(input)
    print("Block 1: ", output1.shape)

    model = Block2()
    output2 = model(maxpool(output1))
    print("Block 2: ", output2.shape)

    model = Block3()
    output3 = model(maxpool(output2))
    print("Block 3: ", output3.shape)

    model = Block4()
    output4 = model(maxpool(output3))
    print("Block 4: ", output4.shape)

    model = Block5()
    output5 = model(maxpool(output4))
    print("Block 5: ", output5.shape)

    model = Encoder()
    output, _ = model(input)
    print("Encoder: ", output.shape)

    model = UpBlock4()
    output = model(output, output4)
    print("Up block 4: ", output.shape)

    model = UpBlock3()
    output = model(output, output3)
    print("Up block 3: ", output.shape)

    model = UpBlock2()
    output = model(output, output2)
    print("Up block 2: ", output.shape)

    model = UpBlock1()
    output = model(output, output1)
    print("Up block 1: ", output.shape)

    model = Encoder()
    output, r = model(input)
    model = Decoder()
    output = model(output, r)
    print("Decoder: ", output.shape)

    model = UNet()
    output = model(input)
    print("UNet: ", output.shape)
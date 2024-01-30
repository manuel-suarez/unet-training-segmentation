# Own scratch implementation
import torch
import torch.nn as nn
import unittest

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        return x

# Encoder block
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = EncoderBlock(in_channels=3, out_channels=64)
        self.block2 = EncoderBlock(in_channels=64, out_channels=128)
        self.block3 = EncoderBlock(in_channels=128, out_channels=256)
        self.block4 = EncoderBlock(in_channels=256, out_channels=512)
        self.block5 = EncoderBlock(in_channels=512, out_channels=1024)
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

class DecoderUpBlock(nn.Module):
    def __init__(self, upsize, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.upsampling = nn.Upsample(size=(upsize, upsize))
        self.upconv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2)
        # concatenation
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
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

class UpBlock4(DecoderUpBlock):
    def __init__(self):
        super().__init__(upsize=57, in_channels=1024, out_channels=512)

class UpBlock3(DecoderUpBlock):
    def __init__(self):
        super().__init__(upsize=105, in_channels=512, out_channels=256)

class UpBlock2(DecoderUpBlock):
    def __init__(self):
        super().__init__(upsize=201, in_channels=256, out_channels=128)

class UpBlock1(DecoderUpBlock):
    def __init__(self):
        super().__init__(upsize=393, in_channels=128, out_channels=64)

class SegmentationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)
    def forward(self, x):
        x = self.conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block4 = UpBlock4()
        self.block3 = UpBlock3()
        self.block2 = UpBlock2()
        self.block1 = UpBlock1()
        self.head = SegmentationHead()

    def forward(self, x, r):
        x = self.block4(x, r[3])
        x = self.block3(x, r[2])
        x = self.block2(x, r[1])
        x = self.block1(x, r[0])
        x = self.head(x)

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


class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.input_channels = 3
        self.input_size = 572
        self.block_channels = [64, 128, 256, 512, 1024]

    def test_encoder_block1(self):
        block_size = self.input_size
        input = torch.randn((1, self.input_channels, block_size, block_size))
        output = Encoder().block1(input)
        self.assertEqual(output.shape, (1, self.block_channels[0], block_size - 4, block_size - 4))

    def test_encoder_block2(self):
        block_size = (self.input_size - 4) // 2
        input = torch.randn((1, self.block_channels[0], block_size, block_size))
        output = Encoder().block2(input)
        self.assertEqual(output.shape, (1, self.block_channels[1], block_size - 4, block_size - 4))

    def test_encoder_block3(self):
        block_size = (((self.input_size - 4) // 2) - 4) // 2
        input = torch.randn((1, self.block_channels[1], block_size, block_size))
        output = Encoder().block3(input)
        self.assertEqual(output.shape, (1, self.block_channels[2], block_size - 4, block_size - 4))

    def test_encoder_block4(self):
        block_size = (((((self.input_size - 4) // 2) - 4) // 2) - 4) // 2
        input = torch.randn((1, self.block_channels[2], block_size, block_size))
        output = Encoder().block4(input)
        self.assertEqual(output.shape, (1, self.block_channels[3], block_size - 4, block_size - 4))

    def test_encoder_block5(self):
        block_size = (((((((self.input_size - 4) // 2) - 4) // 2) - 4) // 2) - 4) // 2
        input = torch.randn((1, self.block_channels[3], block_size, block_size))
        output = Encoder().block5(input)
        self.assertEqual(output.shape, (1, self.block_channels[4], block_size - 4, block_size - 4))

    def test_encoder_blocks(self):
        input = torch.randn((1, self.input_channels, self.input_size, self.input_size))
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        output1 = Encoder().block1(input)
        block_size = self.input_size - 4
        self.assertEqual(output1.shape, (1, self.block_channels[0], block_size, block_size))
        output2 = Encoder().block2(maxpool(output1))
        block_size = block_size // 2 - 4
        self.assertEqual(output2.shape, (1, self.block_channels[1], block_size, block_size))
        output3 = Encoder().block3(maxpool(output2))
        block_size = block_size // 2 - 4
        self.assertEqual(output3.shape, (1, self.block_channels[2], block_size, block_size))
        output4 = Encoder().block4(maxpool(output3))
        block_size = block_size // 2 - 4
        self.assertEqual(output4.shape, (1, self.block_channels[3], block_size, block_size))
        output5 = Encoder().block5(maxpool(output4))
        block_size = block_size // 2 - 4
        self.assertEqual(output5.shape, (1, self.block_channels[4], block_size, block_size))

    def test_encoder_module(self):
        input = torch.randn((1, 3, 572, 572))
        output, _ = Encoder()(input)
        self.assertEqual(output.shape, (1, 1024, 28, 28))

class TestDecoder(unittest.TestCase):
    def test_decoder_upblock4(self):
        input = torch.randn((1, 1024, 28, 28))
        skip = torch.rand((1, 512, 64, 64))
        output = UpBlock4()(input, skip)
        self.assertEqual(output.shape, (1, 512, 52, 52))

    def test_decoder_upblock3(self):
        input = torch.randn((1, 512, 64, 64))
        skip = torch.rand((1, 256, 136, 136))
        output = UpBlock3()(input, skip)
        self.assertEqual(output.shape, (1, 256, 100, 100))

    def test_decoder_upblock2(self):
        input = torch.randn((1, 256, 100, 100))
        skip = torch.rand((1, 128, 280, 280))
        output = UpBlock2()(input, skip)
        self.assertEqual(output.shape, (1, 128, 196, 196))

    def test_decoder_upblock1(self):
        input = torch.randn((1, 128, 196, 196))
        skip = torch.rand((1, 64, 568, 568))
        output = UpBlock1()(input, skip)
        self.assertEqual(output.shape, (1, 64, 388, 388))

    def test_segmentation_head(self):
        input = torch.randn((1, 64, 388, 388))
        output = SegmentationHead()(input)
        self.assertEqual(output.shape, (1, 2, 388, 388))

    def test_decoder_blocks(self):
        input = torch.randn((1, 3, 572, 572))
        output, skips = Encoder()(input)

        output = UpBlock4()(output, skips[3])
        self.assertEqual(output.shape, (1, 512, 52, 52))
        output = UpBlock3()(output, skips[2])
        self.assertEqual(output.shape, (1, 256, 100, 100))
        output = UpBlock2()(output, skips[1])
        self.assertEqual(output.shape, (1, 128, 196, 196))
        output = UpBlock1()(output, skips[0])
        self.assertEqual(output.shape, (1, 64, 388, 388))
        output = SegmentationHead()(output)
        self.assertEqual(output.shape, (1, 2, 388, 388))

    def test_decoder_module(self):
        input = torch.randn((1, 3, 572, 572))
        output, r = Encoder()(input)
        output = Decoder()(output, r)
        self.assertEqual(output.shape, (1, 2, 388, 388))

class TestUNet(unittest.TestCase):
    def test_unet(self):
        model = UNet()
        input = torch.randn((1, 3, 572, 572))
        output = model(input)
        self.assertEqual(output.shape, (1, 2, 388, 388))

class UNetTestSuite(unittest.TestSuite):
    def __init__(self):
        super(self).__init__()
        self.addTest(TestEncoder())
        self.addTest(TestDecoder())

if __name__ == '__main__':
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    model = Block1()
    input = torch.randn((1, 3, 572, 572))
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

    unittest.main()
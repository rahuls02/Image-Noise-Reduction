from styleGAN import Block
import torch

BatchSize = 32
ColorSpace = 3
width = height = size =  4


def test_block_sizes():
    block = Block(size, upsample=False)
    # Batch size x color space x width x height
    noise = torch.randn(BatchSize, ColorSpace, size, size)
    styleNoise = torch.randn(1, 512)
    randNoise = torch.randn(1, 512)
    result = block(noise, styleNoise, randNoise)
    assert result.Size == (BatchSize, ColorSpace, size, size)

def test_block_sizes_upsample():
    block = Block(size, upsample=True)
    # Batch size x color space x width x height
    noise = torch.randn(BatchSize, ColorSpace, size, size)
    styleNoise = torch.randn(1, 512)
    randNoise = torch.randn(1, 512)
    randNoise = torch.randn(1, 512)
    assert result.Size == (BatchSize, ColorSpace, size*2, size*2)

test_block_sizes()
test_block_sizes_upsample()

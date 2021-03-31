from torch import nn
import torch
from .shared import quick_scale


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        # Convert image from rgb image to "latent_space" image
        # start with a resolution of 16 x 16 and scale upwards
        self.from_rgbs = nn.ModuleList([
            quick_scale(nn.Conv2d(3, 16,  1)),
            quick_scale(nn.Conv2d(3, 32,  1)),
            quick_scale(nn.Conv2d(3, 64,  1)),
            quick_scale(nn.Conv2d(3, 128, 1)),
            quick_scale(nn.Conv2d(3, 256, 1)),
            quick_scale(nn.Conv2d(3, 512, 1)),
            quick_scale(nn.Conv2d(3, 512, 1)),
            quick_scale(nn.Conv2d(3, 512, 1)),
            quick_scale(nn.Conv2d(3, 512, 1))
        ])

        self.convs = nn.ModuleList([
            UpscaleBlock(16, 32,   kernel1=3, padding1=1),
            UpscaleBlock(32, 64,   kernel1=3, padding1=1),
            UpscaleBlock(64, 128,  kernel1=3, padding1=1),
            UpscaleBlock(128, 256, kernel1=3, padding1=1),
            UpscaleBlock(256, 512, kernel1=3, padding1=1),
            UpscaleBlock(512, 512, kernel1=3, padding1=1),
            UpscaleBlock(512, 512, kernel1=3, padding1=1),
            UpscaleBlock(512, 512, kernel1=3, padding1=1),
            UpscaleBlock(513, 512, kernel1=3, padding1=1, kernel2=4, padding2=0)
        ])
        self.layer_count = len(self.convs)
        self.fc = quick_scale(nn.Linear(512, 1))

    def forward(self, image,
            step = 0,
            alpha = -1):

        for i in range(step, -1, -1):
            layer_index = self.layer_count - i - 1

            # First layer, convert from rgb to n_channel data
            if i == step:
                result = self.from_rgbs[layer_index](image)

            # Last layer, minibatch std
            if i == 0:
                # In dim: [batch, channel(512), 4, 4]
                res_var = result.var(0, unbiased=False) + 1e-8  # avoid 0
                # Out dim: [channel(512), 4, 4]
                res_std = torch.sqrt(res_var)
                # Out dim: [channel(512), 4, 4]
                mean_std = res_std.mean().expand(result.size(0), 1, 4, 4)
                # Outdim: [1] -> [batch, 1, 4, 4]
                result = torch.cat([result, mean_std], 1) # Add mean std
                # out dim: [batch, 512 +1, 4, 4]
            result = self.convs[layer_index](result)

            if i > 0:
                # Downsample
                result = nn.functional.interpolate(result,
                                                   scale_factor=0.5,
                                                   mode='bilinear',
                                                   recompute_scale_factor=True,
                                                   align_corners=False)

                if i == step and 0 <= alpha <1:
                    result_next = self.from_rgbs[layer_index + 1](image)
                    result_next = nn.functional.interpolate(result_next, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
                    result = alpha * result + (1 - alpha) * result_next

        # Convert from [batch, channel(512), 1, 1] -> [batch, channel(512)]
        result = result.squeeze(2).squeeze(2)
        return self.fc(result)

class UpscaleBlock(nn.Module):

    def __init__(self, in_channel, out_channel, *, kernel1, padding1, kernel2=None, padding2=None):
        super().__init__()
        if kernel2 == None:
            kernel2 = kernel1
        if padding2 == None:
            padding2 = padding1

        self.conv = nn.Sequential(
            quick_scale(nn.Conv2d(in_channel, out_channel, kernel1, padding=padding1)),
            nn.LeakyReLU(0.2),
            quick_scale(nn.Conv2d(out_channel, out_channel, kernel2, padding=padding2)),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)

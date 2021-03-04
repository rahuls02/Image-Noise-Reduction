from torch import nn
import torch
import numpy as np
from torch.nn import functional as func
from .shared import quick_scale

"""
Style Based generator, design is heavily based on:
https://github.com/SiskonEmilia/StyleGAN-PyTorch/blob/c11e94e51478d5134729936066f1242488e9e6ee/model.py
"""

class StyleBasedGenerator(nn.Module):

    def __init__(self, out_channels, *, num_fc, dim_latent, dim_input):
        self.fcs = LatentMapper(dim_latent, num_layers=num_fc)
        self.convs = nn.ModuleList([
            FirstBlock(512, dim_latent, dim_input),
            GeneratorBlock(512, 512, dim_latent=dim_latent),
            GeneratorBlock(512, 512, dim_latent=dim_latent),
            GeneratorBlock(512, 512, dim_latent=dim_latent),
            GeneratorBlock(512, 256, dim_latent=dim_latent),
            GeneratorBlock(256, 128, dim_latent=dim_latent),
            GeneratorBlock(128, 64,  dim_latent=dim_latent),
            GeneratorBlock(64, 32,   dim_latent=dim_latent),
            GeneratorBlock(32, 16,   dim_latent=dim_latent)
        ])

        self.to_rgbs = nn.ModuleList([
            quick_scale(nn.Conv2d(512, 3, 1)),
            quick_scale(nn.Conv2d(512, 3, 1)),
            quick_scale(nn.Conv2d(512, 3, 1)),
            quick_scale(nn.Conv2d(512, 3, 1)),
            quick_scale(nn.Conv2d(256, 3, 1)),
            quick_scale(nn.Conv2d(128, 3, 1)),
            quick_scale(nn.Conv2d(64 , 3, 1)),
            quick_scale(nn.Conv2d(32 , 3, 1)),
            quick_scale(nn.Conv2d(16 , 3, 1))
        ])

        def forward(self, latent_z,
                step = 0, # The current is how many layers away from 4 x 4
                alpha = -1, # Smooth conversion (upscaling / downscaling)
                noise = None,
                mix_steps = [],
                latent_w_center = None,
                psi = 0):
            
            if type(latent_z) != type([]):
                print("Please use a list to package latent_z")
                latent_z = [latent_z]
            if (len(latent_z) != 2 and len(mix_steps) > 0) or type(mix_steps) != type([]):
                print('Warning: Style mixing disabled, possible reasons:')
                print('- Invalid number of latent vectors')
                print('- Invalid parameter type: mix_steps')
                mix_steps = []

            latent_w = [self.fcs(latent) for latent in latent_z]
            batch_size = latent_w[0].size(0)

            # Truncation trick in W
            if latent_w_center is not None:
                latent_w = [latent_w_center + psi * (unscaled_latenet_w - latent_w_center)
                        for unscaled_latenet_w in latent_w]

            result = 0
            current_latent = 0
            for i, conv in enumerate(self.convs):
                if i in mix_steps:
                    current_latent = latent_w[1]
                else:
                    current_latent = latent_w[0]

                if i > 0 and step > 0:# Should we be upsampling in this layer
                    result_upsample = nn.functional.interpolate(result, scale_factor=2, mode='bilinear',
                                                      align_corners=False)
                    result = conv(result_upsample, current_latent, noise[i])
                else:
                    result = conv(current_latent, noise[i])

                # Final layer
                if i == step:
                    # Could stop early
                    result = self.to_rgbs[i](result)
                    if i > 0 and 0 <= alpha < 1:
                        result_prev = self.to_rgbs[i - 1](result_upsample)
                        result = alpha * result + ( 1 - alpha) * result_prev
                    break

            return result

class FirstBlock(nn.Module):

    def __init__(self, in_size, dim_latent, dim_input):
        super().__init__()

        self.constant = nn.Parameter(torch.randn(1, in_size, dim_input, dim_input))

        self.conv = quick_scale(nn.Conv2d(in_size, in_size, padding=1, kernel_size=3)),
        # Noise Mappers
        self.noise_layer1 = quick_scale(NoiseLayer(in_size))
        self.noise_layer2 = quick_scale(NoiseLayer(in_size))
        # Style Mappers
        self.style1 = StyleLayer(dim_latent, in_size)
        self.style2 = StyleLayer(dim_latent, in_size)
        # Normalization
        self.adain = AdaIn()
        # Activation
        self.act = nn.LeakyReLU(0.2)

    def forward(self, style, noise):

        x = self.constant.repeat(noise.shape[0], 1, 1, 1)
        x = x + self.noise_layer1(noise)
        x = self.adain(x, self.style1(style))
        x = self.act(x)
        x = self.conv(x)
        x = x + self.noise_layer2(x)
        x = self.adain(x, self.style2(style))
        x = self.act(x)

        return x



class GeneratorBlock(nn.Module):

    def __init__(self, in_size, out_size, *, dim_latent): # Assume square
        super().__init__()

        self.conv1 = quick_scale(nn.Conv2d(in_size, in_size, padding=1, kernel_size=3)),
        self.conv2 = quick_scale(nn.Conv2d(in_size, in_size, padding=1, kernel_size=3)),
        # Noise Mappers
        self.noise_layer1 = quick_scale(NoiseLayer(out_size))
        self.noise_layer2 = quick_scale(NoiseLayer(out_size))
        # Style Mappers
        self.style1 = StyleLayer(dim_latent, out_size)
        self.style2 = StyleLayer(dim_latent, out_size)
        # Normalization
        self.adain = AdaIn()
        # Activation
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, style, noise):
        x = self.conv1(x)
        x = x + self.noise_layer1(noise) # Adding bias (noise)
        x = self.adain(x, self.style1(style))
        x = self.act(x)
        x = self.conv2(x)
        x = x + self.noise_layer2(noise) # Adding bias (noise)
        x = self.adain(x, self.style2(style))
        x = self.act(x)
        return x

class PixelNorm(nn.Module):
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=1) + 1e-8)

class LatentMapper(nn.Module):

    def __init__(self, z_dim_size, num_layers=8):
        """
        Maps the Z vector space to the W vector space
        """
        super().__init__()
        # 512 x 1
        layers = [PixelNorm()]
        for i in range(num_layers):
            layers.append(quick_scale(nn.Linear(z_dim_size, z_dim_size)))
            layers.append(nn.LeakyReLU(0.2))

        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        return self.mapping(x)

class NoiseLayer(nn.Module):
    """
    Simply scales the noise weight
    """
    def __init__(self, n_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, n_channels, 1, 1)) # initialize weights to 0 initially

    def forward(self, noise):
        result = noise * self.weight
        return result

class StyleLayer(nn.Module):

    def __init__(self, z_dim, n_channels):
        linear = nn.Linear(z_dim, n_channels * 2)

        linear.transform.linear.bias.data[:n_channels] = 1
        linear.transform.linear.bias.data[n_channels:] = 0

        self.transform = quick_scale(linear)

    def forward(self, latent):
        return self.transform(latent).unsqueeze(2).unsqueeze(3)


# AdaIn taken from this github 
# https://github.com/irasin/Pytorch_AdaIN/blob/2882d729d46d5d61048b6ed7893505bafdc2abc3/model.py Citation from 
class AdaIn(nn.Module):

    def calc_mean_std(self, features):
        """
        :param features: shape of features -> [batch_size, c, h, w]
        :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
        """

        batch_size, c = features.size()[:2]
        features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
        features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
        return features_mean, features_std

    def forward(self, content_features, style_features):
        """
        Adaptive Instance Normalization

        :param content_features: shape -> [batch_size, c, h, w]
        :param style_features: shape -> [batch_size, c, h, w]
        :return: normalized_features shape -> [batch_size, c, h, w]
        """
        content_mean, content_std = self.calc_mean_std(content_features)
        style_mean, style_std = self.calc_mean_std(style_features)
        normalized_features = style_std * (content_features - content_mean) / content_std + style_mean

        return normalized_features
#

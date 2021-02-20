from torch import nn
from torch.nn import functional as func

class StyleGAN(nn.Module):

    def __init__(self, out_channels):
        # 10 blocks
        pass

class Block(nn.Module):

    def __init__(self, in_size, upsample=False): # Assume square
        super().__init__()
        self.adain = AdaIn()

        self.conv1 = nn.Linear(
                nn.Conv2d(in_size, in_size, padding=1, kernel_size=3, bias=False)
                nn.LeakyReLU(0.2)
                )
        self.conv2 = nn.Linear(
                nn.Conv2d(in_size, in_size, padding=1, kernel_size=3, bias=False)
                nn.LeakyReLU(0.2)
                )

        self.upsample = upsample

    def forward(self, x, style, noise):
        if self.upsample:
            x = func.upsample(x, scale_factor=2)
        # x = inject_noise(noise_b)
        x = self.conv1(x)
        x = x + self.noise # TODO: Add weight to this noise
        x = self.conv2(x)
        x = self.adain(x, style_noise)
        return x


class LatentMapper(nn.Module):

    def __init__(self, z_dim_size):
        super().__init__()
        # 512 x 1
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(z_dim_size, z_dim_size))

        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        return self.mapping(x)


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

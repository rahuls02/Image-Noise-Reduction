from torch import nn
import numpy as np

class Scaler:
    """
         Scales the input modules weights for upscaling
    """
    def __init__(self, name):
        self.name = name

    def scale(self, module):
        weight = getattr(module, self.name + '_base')
        fan_in = weight.data.size(1) + weight.data[0][0].numel()

        return weight * np.sqrt(2 / fan_in)

    def __call__(self, module, input):
        weight = self.scale(module)
        setattr(module, self.name, weight)  # Sets module.weight to the new scaled weight

    @staticmethod
    def apply_scale(module: nn.Module, name="weight"):
        hook = Scaler(name)
        weight = getattr(module, name)
        module.register_parameter(name + '_base', nn.Parameter(weight.data))
        del module._parameters[name]  # Lets save up some space
        module.register_forward_pre_hook(hook)  # makes the Scaler be called every time forward() is called

def quick_scale(module, name="weight"):
    """
    Adds a hook to easily upscale / downscale the input
    """
    Scaler.apply_scale(module, name)
    return module

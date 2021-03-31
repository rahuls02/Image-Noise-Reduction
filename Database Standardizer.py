#!/usr/bin/env python
# coding: utf-8
from PIL import Image
from torch.utils.data import Dataset
from typing import List
import numpy as np
import os

# credit to https://gist.github.com/glenrobertson/2288152#gistcomment-3461365
def get_white_noise_image(w,h):
    pil_map = Image.fromarray(np.random.randint(0,255,(w,h,3),dtype=np.dtype('uint8')))
    return pil_map

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image(filename:str): 
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(root_dir:str)-> List[str]:
    images = []
    for root, _, fnames in sorted(os.walk(root_dir)):
        for fname in fnames:
            if is_image(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

class NoiseImageDataset(Dataset):

    def __init__(self, root, noise_strength=0.2, transform=None):
        self.root = root
        self.paths = make_dataset(root)
        self.noise_strength = noise_strength
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        clean_im = Image.open(path)
        clean_im = clean_im.convert('RGB')

        if self.transform:
            clean_im = self.transform(clean_im)
        
        # Noise it up
        noised_im = Image.blend(clean_im, get_white_noise_image(clean_im.size()[0], clean_im.size()[1]), self.noise_strength)
        return clean_im, noised_im

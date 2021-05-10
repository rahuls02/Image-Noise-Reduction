from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from utils import data_utils

# credit to https://gist.github.com/glenrobertson/2288152#gistcomment-3461365


def get_white_noise_image(w, h):
    pil_map = Image.fromarray(
        np.random.randint(0, 255, (w, h, 3), dtype=np.dtype("uint8"))
    )
    return pil_map.convert("RGB")


class NoiseImagesDataset(Dataset):
    """
    Image dataset to apply noise to the files in source_root onto, and applies source_transform onto
    both images.
    """

    def __init__(
        self,
        source_root: str,
        target_root: str,
        opts,
        target_transform=None,
        source_transform=None,
    ):
        self.source_paths = sorted(data_utils.make_dataset(source_root))
        self.noise_strength = opts.noise_strength
        self.transform = source_transform
        self.opts = opts

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        path = self.source_paths[index]
        clean_im = Image.open(path)  # Noise it up
        noised_im = Image.blend(
            clean_im,
            get_white_noise_image(clean_im.size[1], clean_im.size[0]),
            self.noise_strength,
        )
        if self.transform:
            clean_im = self.transform(clean_im)
            noised_im = self.transform(noised_im)

        return noised_im, clean_im


class ImagesDataset(Dataset):
    def __init__(
        self,
        source_root,
        target_root,
        opts,
        target_transform=None,
        source_transform=None,
    ):
        self.source_paths = sorted(data_utils.make_dataset(source_root))
        self.target_paths = sorted(data_utils.make_dataset(target_root))
        self.target_transform = target_transform
        self.source_transform = source_transform
        self.opts = opts

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        from_path = self.source_paths[index]
        from_im = Image.open(from_path)
        from_im = (
            from_im.convert("RGB") if self.opts.label_nc == 0 else from_im.convert("L")
        )

        to_path = self.target_paths[index]
        to_im = Image.open(to_path).convert("RGB")
        if self.target_transform:
            to_im = self.target_transform(to_im)

        if self.source_transform:
            from_im = self.source_transform(from_im)
        else:
            from_im = to_im

        return from_im, to_im

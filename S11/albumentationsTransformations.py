import numpy as np
from albumentations.core.composition import Compose
from albumentations.augmentations.transforms import VerticalFlip
from albumentations.augmentations.transforms import HorizontalFlip
from albumentations.augmentations.transforms import HueSaturationValue
from albumentations.augmentations.transforms import RandomSizedCrop
from albumentations.augmentations.transforms import RandomCrop
from albumentations.augmentations.transforms import Rotate
from albumentations.augmentations.transforms import PadIfNeeded
from albumentations.augmentations.transforms import Cutout
from albumentations.pytorch.transforms import ToTensor
from albumentations.augmentations.transforms import Normalize

class AlbumentationsTransforms:
    def __init__(self):
        self.transforms = Compose([
       PadIfNeeded(40,40),
       RandomCrop(32,32),
       HorizontalFlip(p=.5),
       Cutout(8, 8),
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), max_pixel_value=255.0, always_apply=True, p=1.0),
        ToTensor()
    ])
    
    def __call__(self, img):
        img = np.array(img)
        img = self.transforms(image=img)['image']
        return img
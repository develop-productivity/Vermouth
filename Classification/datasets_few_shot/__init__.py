"""copy from Tip-Adapter"""

from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .imagenet import ImageNet

import os
DATASET_ROOT = os.getenv('DATASET_ROOT', '/data/sydong/datasets')

dataset_list = {
                "oxford_pets": OxfordPets,
                "eurosat": EuroSAT,
                "ucf101": UCF101,
                "sun397": SUN397,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "aircraft": FGVCAircraft,
                "food101": Food101,
                "flowers": OxfordFlowers,
                "stanford_cars": StanfordCars,
                "imagenet": ImageNet,
                }


def build_dataset(dataset, shots, process=None):
    root_path = DATASET_ROOT
    if dataset == 'imagenet':
        assert process is not None
        return dataset_list[dataset](shots, process)
    else:
        return dataset_list[dataset](root_path, shots)
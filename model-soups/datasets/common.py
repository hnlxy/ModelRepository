import os
import torch
import json
import glob
import collections
import random
import math
import time

import numpy as np
from tqdm import tqdm

import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import UnidentifiedImageError


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


class ImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, path, transform):
        super().__init__(path, transform)

    def __getitem__(self, index):

        max_attempts = 5
        attempts = 0
        while attempts < max_attempts:
            try:
                image, label = super(ImageFolderWithPaths, self).__getitem__(index)
                return {
                    'images': image,
                    'labels': label,
                    'image_paths': self.samples[index][0]
                }
            except (UnidentifiedImageError, OSError, IndexError) as e:
                print(f"[Warning] Skipping image at index {index} due to error: {e}")
                index = (index + 1) % len(self.samples)
                attempts += 1
        raise RuntimeError(f"Too many corrupted images encountered starting from index {index}")


import glob

import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png")

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)





class cityscapes(Dataset):
    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root,'leftImg8bit', subset)
        self.labels_root = os.path.join(root, 'gtFine', subset)

        print(f"Images path: {self.images_root}")
        print(f"Labels path: {self.labels_root}")

        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(self.images_root) for f in fn if is_image(f)]
        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(self.labels_root) for f in fn if is_label(f)]

        self.filenames.sort()
        self.filenamesGt.sort()

        print(f"Number of image files: {len(self.filenames)}")
        print(f"Number of label files: {len(self.filenamesGt)}")

        print("Sample image filenames:")
        for f in self.filenames[:5]:
            print(f)
        print("Sample label filenames:")
        for f in self.filenamesGt[:5]:
            print(f)

        # Filtra per trovare coppie valide
        valid_filenames = []
        valid_filenamesGt = []

        for f in self.filenames:
            base_name = os.path.basename(f).replace("_leftImg8bit.png", "")
            for gt in self.filenamesGt:
                gt_base_name = os.path.basename(gt).replace("_gtFine_labelTrainIds.png", "")
                if base_name == gt_base_name:
                    valid_filenames.append(f)
                    valid_filenamesGt.append(gt)
                    break

        self.filenames = valid_filenames
        self.filenamesGt = valid_filenamesGt

        print(f"Number of valid image-label pairs: {len(self.filenames)}")
        if len(self.filenames) != len(self.filenamesGt):
            print("Warning: Mismatch in number of images and labels after filtering!")

        self.co_transform = co_transform

    def __getitem__(self, index):
        if index >= len(self.filenames) or index >= len(self.filenamesGt):
            raise IndexError(f"Index {index} out of range. Check dataset alignment.")

        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(filename, 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(filenameGt, 'rb') as f:
            label = load_image(f).convert('P')

        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)

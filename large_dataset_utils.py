# Copyright (c) 2023-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from imagenet_classnames import folder_label_map

from torchvision.datasets.stanford_cars import StanfordCars
from torchvision.datasets.food101 import Food101

# Reverse dictionary mapping
label_folder_map = {v: k for k, v in folder_label_map.items()}

cur_file_path = os.path.dirname(os.path.abspath(__file__))

class ImageNetSubset_Dataset(Dataset):
    
    def __init__(self, root, class_names, override_count=False, nImages_per_class=1300, selected_image_list_path=None, transform=None):
        """
            override_count: Specificies the maximum number of samples per class (now maximum nImages_per_clas number of images per class)
        """
        self.img_path = []
        self.labels = []
        self.transform = transform


        nClasses = len(class_names)

        if selected_image_list_path is not None: # Load the image paths
            image_list_path = os.path.join(selected_image_list_path, f"{nClasses}", "image_paths.txt")
            selected_image_paths = []
            with open(image_list_path, "r") as f:
                selected_image_paths = f.readlines()

            selected_image_paths = list(map(lambda x : x.strip(), selected_image_paths))
            selected_image_paths = set(selected_image_paths)
        
        for i, name in enumerate(class_names):
            folder_name = name
            folder_path = os.path.join(root, folder_name)
            sample_count = 0 # Counter of the number of images for this class
            for fid in os.listdir(folder_path):
                if override_count and sample_count + 1 > nImages_per_class:
                    break
                file_path = os.path.join(folder_path, fid)
                if selected_image_list_path is not None: 
                    # Ensure the image to load is on the list of selected images
                    if file_path not in selected_image_paths:
                        continue
                    
                self.img_path.append(file_path)
                self.labels.append(i)
                sample_count += 1

        self.targets = self.labels # Sampler needs to use targets
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index




class ImageNet100_Dataset(Dataset):
    
    def __init__(self, root, transform=None, split=None):
        self.img_path = []
        self.labels = []
        self.transform = transform

        with open(os.path.join(cur_file_path, 'imagenet100.txt')) as f:  # The class subset is taken from: https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt
            class_names = list(map(lambda x : x.strip(), f.readlines()))
        nClasses = 100 # Subset of ImageNet
        for i, name in enumerate(class_names):
            folder_name = name
            folder_path = os.path.join(root, folder_name)

            file_names = os.listdir(folder_path)
            if split is not None:
                num_train = int(len(file_names) * 0.8) # 80% Training data
            for j, fid in enumerate(file_names):
                if split == 'train' and j >= num_train: # ensures only the first 80% of data is used for training
                    break
                elif split == 'test' and j < num_train: # skips the first 80% of data used for training
                    continue
                self.img_path.append(os.path.join(folder_path, fid))
                self.labels.append(i)
        print(f"Dataset Size: {len(self.labels)}")
        self.targets = self.labels # Sampler needs to use targets
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index


class ImageNet_Dataset(Dataset):
    
    def __init__(self, root, transform=None, split=None):
        self.img_path = []
        self.labels = []
        self.transform = transform

        nClasses = 1000 # Subset of ImageNet
        all_class_names = sorted(os.listdir(root))
        for i, name in enumerate(all_class_names):
            folder_name = name
            folder_path = os.path.join(root, folder_name)

            file_names = os.listdir(folder_path)
            if split is not None:
                num_train = int(len(file_names) * 0.8) # 80% Training data
            for j, fid in enumerate(file_names):
                if split == 'train' and j >= num_train: # ensures only the first 80% of data is used for training
                    break
                elif split == 'test' and j < num_train: # skips the first 80% of data used for training
                    continue
                self.img_path.append(os.path.join(folder_path, fid))
                self.labels.append(i)
        print(f"Dataset Size: {len(self.labels)}")
        self.targets = self.labels # Sampler needs to use targets
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return np.asarray(sample), label, index


class Cars(StanfordCars):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Cars, self).__init__(root, split='train' if train else 'test', transform=transform,
                                     target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


class Food(Food101):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Food, self).__init__(root, split='train' if train else 'test', transform=transform,
                                     target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

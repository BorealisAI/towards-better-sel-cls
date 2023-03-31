# MIT License

# Copyright (c) 2021 Layne

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision import transforms, utils, datasets
from PIL import Image

class resized_dataset(Dataset):
    def __init__(self, dataset, transform=None, start=None, end=None, resize=None):
        self.data=[]
        if start == None:
            start = 0
        if end == None:
            end = dataset.__len__()
        if resize is None:
            for i in range(start, end):
                self.data.append((*dataset.__getitem__(i)))
        else:
            for i in range(start, end):
                item = dataset.__getitem__(i)
                self.data.append((F.center_crop(F.resize(item[0],resize,Image.BILINEAR), resize), item[1]))
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.transform:
            return (self.transform(self.data[idx][0]), self.data[idx][1], idx)
        else:
            return self.data[idx], idx


class C10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(C10, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index


class SVHN(datasets.SVHN):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(SVHN, self).__init__(root, split=split, transform=transform,
                                     target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class CatsDogs(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None, resize=None):
        super(CatsDogs, self).__init__()
        self.root = os.path.join(root, "train")
        self.resize  = resize
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        with open(os.path.join(root, split + "_gambler_split.txt"), 'r') as fin:
            for fname in fin.readlines():
                self.data.append(fname.strip())

    def __getitem__(self, index):
        fname = self.data[index]
        
        # read and scale image
        img = Image.open(os.path.join(self.root, fname))
        if self.resize is not None:
            img = F.center_crop(F.resize(img, self.resize, Image.BILINEAR), self.resize)

        # obtain label
        target = 0 if fname.split('.')[0] == 'cat' else 1

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    def __len__(self):
        return len(self.data)


from imagenet_classnames import name_map, folder_label_map

# Create a reverse dictionary that maps items to keys
label_folder_map = dict(list(map(lambda z : (z[1], z[0]), folder_label_map.items()))) 

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



cur_file_path = os.path.dirname(os.path.abspath(__file__))


with open(os.path.join(cur_file_path, 'imagenet100.txt')) as f:  # The class subset is taken from: https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt
    class_names = list(map(lambda x : x.strip(), f.readlines()))

class ImageNet100_Dataset(Dataset):
    
    def __init__(self, root, transform=None, split=None):
        self.img_path = []
        self.labels = []
        self.transform = transform

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

        # return sample, label, path
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





import pathlib
from typing import Any, Callable, Optional, Tuple

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset



class StanfordCars(datasets.VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split into 8,144 training images and 8,041 testing images, where each class
    has been split roughly in a 50-50 split

    .. note::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path("/home/leofeng/datasets/cars") / "stanford_cars"
        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = self._base_folder / "cars_test_annos_withlabels.mat"
            self._images_base_path = self._base_folder / "cars_test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        ]

        self.classes = sio.loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self._samples)


    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target


    def download(self) -> None:
        if self._check_exists():
            return

        download_and_extract_archive(
            url="https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
            download_root=str(self._base_folder),
            md5="c3b158d763b6e2245038c8ad08e45376",
        )
        if self._split == "train":
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
                download_root=str(self._base_folder),
                md5="065e5b463ae28d29e77c1b4b166cfe61",
            )
        else:
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
                download_root=str(self._base_folder),
                md5="4ce7ebf6a94d07f1952d94dd34c4d501",
            )
            download_url(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
                root=str(self._base_folder),
                md5="b0a2b23655a3edd16d84508592a98d10",
            )

    def _check_exists(self) -> bool:
        if not os.path.exists(self._base_folder / "devkit"):
            return False

        return self._annotations_mat_path.exists() and self._images_base_path.is_dir()

class Cars(StanfordCars):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Cars, self).__init__(root, split='train' if train else 'test', transform=transform,
                                     target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


import json
from pathlib import Path
import PIL


class Food101(datasets.VisionDataset):
    """`The Food-101 Data Set <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>`_.

    The Food-101 is a challenging data set of 101 food categories with 101,000 images.
    For each class, 250 manually reviewed test images are provided as well as 750 training images.
    On purpose, the training images were not cleaned, and thus still contain some amount of noise.
    This comes mostly in the form of intense colors and sometimes wrong labels. All images were
    rescaled to have a maximum side length of 512 pixels.


    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    _MD5 = "85eeb15f3717b99a5da872d97d918f87"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = Path(self.root) / "food-101"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "images"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._labels = []
        self._image_files = []
        with open(self._meta_folder / f"{split}.json") as f:
            metadata = json.loads(f.read())

        self.classes = sorted(metadata.keys())
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        for class_label, im_rel_paths in metadata.items():
            self._labels += [self.class_to_idx[class_label]] * len(im_rel_paths)
            self._image_files += [
                self._images_folder.joinpath(*f"{im_rel_path}.jpg".split("/")) for im_rel_path in im_rel_paths
            ]

    def __len__(self) -> int:
        return len(self._image_files)


    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_exists(self) -> bool:
        return all(folder.exists() and folder.is_dir() for folder in (self._meta_folder, self._images_folder))

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self.root, md5=self._MD5)


class Food(Food101):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Food, self).__init__(root, split='train' if train else 'test', transform=transform,
                                     target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

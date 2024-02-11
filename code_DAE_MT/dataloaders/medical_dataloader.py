import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import nibabel as nib
import itertools
from torch.utils.data.sampler import Sampler


def make_dataset(root, mode, num=None):
    """
    Args:
        root (string): dataset directory containing Img & GT folders
        mode (string): 'train', 'val', 'test'
    """
    assert mode in ['train', 'val', 'test']
    items = []

    img_path = os.path.join(root, mode, 'Img')
    mask_path = os.path.join(root, mode, 'GT')

    images = os.listdir(img_path)
    labels = os.listdir(mask_path)
    images.sort()
    labels.sort()

    if num is not None:
        images = images[:num]
        labels = labels[:num]
        print('mode: {} - {} images selected'.format(mode, num))

    for it_im, it_gt in zip(images, labels):
        item = (os.path.join(img_path, it_im), os.path.join(mask_path, it_gt))
        items.append(item)

    return items


def percentile_clip(img_numpy, min_val=0.5, max_val=99.5):
    """
    Intensity normalization based on percentile
    Clips the range based on the quarile values.
    min_val: should be in the range [0,100]
    max_val: should be in the range [0,100]
    """
    low = np.percentile(img_numpy, min_val)
    high = np.percentile(img_numpy, max_val)
    img_numpy[img_numpy < low] = low
    img_numpy[img_numpy > high] = high
    return img_numpy


def normalize_intensity(img_tensor, normalization=None):
    """
    Accept the image tensor and normalizes it (ref: MedicalZooPytorch)
    Args: 
        img_tensor (tensor): image tensor
        normalization (string): choices = "max", "mean"
        norm_values (array, optional): (MEAN, STD, MAX, MIN)
    """
    img_tensor = percentile_clip(img_tensor, 0.5, 99.5)

    if normalization == "mean_percentile":
        mask = img_tensor > np.percentile(img_tensor, 1)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / std_val

    elif normalization == "max":
        img_tensor = img_tensor/img_tensor.max()

    elif normalization == 'full_volume_mean':
        img_tensor = (img_tensor - img_tensor.min()) / img_tensor.max()

    elif normalization == 'max_min':
        img_tensor -= img_tensor.min()
        img_tensor /= img_tensor.max()

    elif normalization == None:
        img_tensor = img_tensor

    return img_tensor


class AbdomenFLARE(Dataset):
    """Abdomen Multi Organ dataset - FLARE."""

    def __init__(self, base_dir=None, split='train', num=None, transform=None, normalization='mean_percentile'):
        """
        Args:
            mode: 'train','val','test'
            base_dir (string): Directory with all the volumes in 'Img' and 'GT' folder.
            transform (callable, optional): Optional transform to be applied on a sample.
            normalization (string, optional): Optional normalization to be applied on volumes. 
        """
        self.root_dir = base_dir
        self.transform = transform
        self.imgs = make_dataset(self.root_dir, split, num)
        self.normalization = normalization

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = nib.load(img_path).get_fdata(dtype=np.float32)
        mask = nib.load(mask_path).get_fdata(dtype=np.float32)
        #img_name = os.path.basename(img_path)
        
        # Normalization
        img = normalize_intensity(img, normalization=self.normalization)
        sample = {'image': img, 'label': mask}

        # same transforms for both image and label
        if self.transform:
            sample = self.transform(sample)

        return sample


class LAHeart(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        if split=='train':
            with open(self._base_dir+'/../train.list', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir+'/../test.list', 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir+"/"+image_name+"/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

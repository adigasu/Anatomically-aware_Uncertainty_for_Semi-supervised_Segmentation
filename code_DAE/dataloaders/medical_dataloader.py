import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import nibabel as nib
import itertools
from torch.utils.data.sampler import Sampler
from skimage.transform import rescale, rotate
from skimage.draw import random_shapes
from skimage.morphology import disk, erosion, dilation, opening, closing


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


class RandomTranslate(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image_t = np.zeros(self.output_size)
        label_t = np.zeros(self.output_size)

        # pad the sample if necessary
        (w, h, d) = label.shape
        ow, oh, od = self.output_size[0], self.output_size[1], self.output_size[2]
        w_start, h_start, d_start = 0, 0, 0
        width, height, depth = w, h, d

        if w > ow:
            w1 = np.random.randint(0, w - ow)
            image = image[w1: w1 + ow, : , :]
            label = label[w1: w1 + ow, : , :]
            width = ow
        elif w < ow:
            w_start = np.random.randint(0, ow - w)
            width = w

        if h > oh:
            h1 = np.random.randint(0, h - oh)
            image = image[:, h1: h1 + oh, :]
            label = label[:, h1: h1 + oh, :]
            height = oh
        elif h < oh:
            h_start = np.random.randint(0, oh - h)
            height = h

        if d > od:
            d1 = np.random.randint(0, d - od)
            image = image[:, :, d1: d1 + od]
            label = label[:, :, d1: d1 + od]
            depth = od
        elif d < od:
            d_start = np.random.randint(0, od - d)
            depth = d

        image_t[w_start : w_start + width , h_start : h_start + height, d_start : d_start + depth] = image
        label_t[w_start : w_start + width , h_start : h_start + height, d_start : d_start + depth] = label

        return {'image': image_t, 'label': label_t}


class RandomRescale(object):
    """
    Resize randomly the image in a sample
    Args:
    p (float, [0,1]): random probability
    """

    def __init__(self, p=0.5, fix_ds=0.0):
        self.prob = p
        self.fix_ds = fix_ds

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if self.fix_ds >= 0.5:
            ds = self.fix_ds
            image = rescale(image, (ds, ds, ds), order=1, mode='constant', cval=0, clip=True, anti_aliasing=True, preserve_range=True)
            label = rescale(label, (ds, ds, ds), order=0, mode='constant', cval=0, clip=True, anti_aliasing=True, preserve_range=True)
            return {'image': image, 'label': label}


        if (np.random.uniform(low = 0.0, high = 1.0) < self.prob):
            fx=np.random.uniform(low=0.8,high=1.1)
            #fy=np.random.uniform(low=0.8,high=1.1) 
            #fz=np.random.uniform(low=0.8,high=1.1) 
            fy = fx
            fz = fx

            image = rescale(image, (fx, fy, fz), order=1, mode='constant', cval=0, clip=True, anti_aliasing=True, preserve_range=True)
            label = rescale(label, (fx, fy, fz), order=0, mode='constant', cval=0, clip=True, anti_aliasing=True, preserve_range=True)

        return {'image': image, 'label': label}


class RandomCorrupt(object):
    """
    Resize randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, p=0.25):
        self.prob = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        noisy_label = label.copy()

        (w, h, c) = label.shape

        # swap pixels, morph, adding artifacts
        for ch in range(c):
            img = noisy_label[:, :, ch]

            '''
            # random artifacts or shapes 
            if (np.random.uniform(low = 0.0, high = 1.0) <= self.prob):
                # Random shapes
                mask, _ = random_shapes((w, h), min_shapes=1, max_shapes=5, max_size=15, allow_overlap=True,
                        channel_axis=None, intensity_range=(0,250))
                img[mask <= 250] = 1
                # Random shapes holes
                mask, _ = random_shapes((w, h), min_shapes=1, max_shapes=5, max_size=15, allow_overlap=True,
                        channel_axis=None, intensity_range=(0,250))
                img[mask <= 250] = 0
            '''

            # swap pixels
            if (np.random.uniform(low = 0.0, high = 1.0) <= self.prob):
                sz = np.random.randint(3, 9)
                footprint = disk(sz)
                indices = np.asarray(np.where(np.abs(img - erosion(img, footprint)) > 0))
                remove_idx = [idx for idx, x in enumerate(indices[0]) if (x + sz >= w)]
                indices = np.delete(indices, remove_idx, 1)
                remove_idx = [idx for idx, x in enumerate(indices[1]) if (x + sz >= h)]
                indices = np.delete(indices, remove_idx, 1)
                indices = indices.astype(int)

                idx = np.random.choice(range(len(indices[0])), size=len(indices[0])//10).astype(int)
                x_l_vals = img[indices[0, idx], indices[1, idx] - sz]
                x_vals = img[indices[0, idx], indices[1, idx]]
                img[indices[0, idx], indices[1, idx] - sz] = x_vals
                img[indices[0, idx], indices[1, idx]] = x_l_vals

                idx = np.random.choice(range(len(indices[0])), size=len(indices[0])//10).astype(int)
                x_l_vals = img[indices[0, idx], indices[1, idx] + sz]
                x_vals = img[indices[0, idx], indices[1, idx]]
                img[indices[0, idx], indices[1, idx] + sz] = x_vals
                img[indices[0, idx], indices[1, idx]] = x_l_vals
                
                idx = np.random.choice(range(len(indices[0])), size=len(indices[0])//10).astype(int)
                x_l_vals = img[indices[0, idx] + sz, indices[1, idx]]
                x_vals = img[indices[0, idx], indices[1, idx]]
                img[indices[0, idx] + sz, indices[1, idx]] = x_vals
                img[indices[0, idx], indices[1, idx]] = x_l_vals

                idx = np.random.choice(range(len(indices[0])), size=len(indices[0])//10).astype(int)
                x_l_vals = img[indices[0, idx] - sz, indices[1, idx]]
                x_vals = img[indices[0, idx], indices[1, idx]]
                img[indices[0, idx] - sz, indices[1, idx]] = x_vals
                img[indices[0, idx], indices[1, idx]] = x_l_vals

            # morph
            if (np.random.uniform(low = 0.0, high = 1.0) <= self.prob):
                morph_flag = np.random.randint(0, 4)
                footprint = disk(np.random.randint(1, 5))
                if morph_flag == 0:
                    img = erosion(img, footprint)
                elif morph_flag == 1:
                    img = dilation(img, footprint)
                elif morph_flag == 2:
                    img = opening(img, footprint)
                elif morph_flag == 3:
                    img = closing(img, footprint)

            noisy_label[:, :, ch] = img

        return {'image': image, 'label': label, 'noisy_label': noisy_label}


class Normalize(object):
    """
    Normalize the image between [0,1]
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image -= image.min()
        image /= image.max()
        
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


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        elif 'noisy_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(), 'noisy_label': torch.from_numpy(sample['noisy_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class BatchSampler(Sampler):
    """Iterate on a sets of indices."""

    def __init__(self, primary_indices, batch_size):
        self.primary_indices = primary_indices
        self.primary_batch_size = batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        return (
            primary_batch
            for (primary_batch)
            in grouper(primary_iter, self.primary_batch_size)
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


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

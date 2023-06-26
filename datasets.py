import numpy as np
import cv2
cv2.setNumThreads(0)
import pickle
import os
import pandas as pd
from sklearn import preprocessing
import glob
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import transforms
import torchvision.transforms as T
from utils.randconv.lib.networks import RandConvModule

from catalyst.data.dataset.torch import DatasetFromSampler
from operator import itemgetter

from PIL import Image
from typing import Iterator, List, Optional, Union


def aug(image, preprocess, severity):
  """Perform AugMix augmentations and compute mixture.
  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.
  Returns:
    mixed: Augmented and mixed image.
  """
  #aug_list = augmentations.augmentations_all
  mixture_width = 3  # Originally it was 3
  mixture_depth = -1
  aug_severity = severity
  if aug_severity < 5:
    from utils.augmix import augmentations
    aug_list = augmentations.augmentations_all
  else:
    #print('Severity = {}'.format(aug_severity))
    from utils.augmix import augmentations1
    aug_list = augmentations1.augmentations_all

  ws = np.float32(np.random.dirichlet([1] * mixture_width))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(image))
  for i in range(mixture_width):
    image_aug = image.copy()
    depth = mixture_depth if mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

################################################################################
def rand_augment(num_ops=5, magnitude=9):
    # This function calls the torchvision.transforms Randaugment function
    # https://pytorch.org/vision/main/generated/torchvision.transforms.RandAugment.html

    # num_ops: No. of operations from a pre-defined list of in-built transformations
    # magnitude: Severity of the chosen transformation
    rand_aug = T.RandAugment(num_ops, magnitude)
    return rand_aug
################################################################################

def get_random_module(net, data_mean, data_std, kernel_sizes, mixing):
    channel_size = 3
    kernel_size = kernel_sizes #[9,11,13,15,17,19]  #7
    mixing = mixing
    identity_prob = 0.0
    rand_bias = True
    distribution = 'kaiming_normal'
    clamp_output = True
    return RandConvModule(net,
                          in_channels=3,
                          out_channels=channel_size,
                          kernel_size=kernel_size,
                          mixing=mixing,
                          identity_prob=identity_prob,
                          rand_bias=rand_bias,
                          distribution=distribution,
                          data_mean=data_mean,
                          data_std=data_std,
                          clamp_output=clamp_output,
                          )


################################################################################
def get_transformers(imsize=64):

    # ImageNet means and std
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    # Initializing Train and Validation transform list
    tt, vt = [], []

    # Train transform
    tt = [transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05)), transforms.Resize((imsize,imsize))]
    tt.extend([transforms.ToTensor(), normalize])

    # Val transform
    vt = [transforms.Resize((imsize,imsize))]
    vt.extend([transforms.ToTensor(), normalize])

    train_trans = transforms.Compose(tt)
    valid_trans = transforms.Compose(vt)

    return train_trans, valid_trans
################################################################################


def get_weighted_sampler(target):
    class_sample_count = np.unique(target, return_counts=True)[1]
    print("Label distribution: {}".format(class_sample_count))
    weights = 1 / torch.Tensor(class_sample_count).float()
    # print("Class weights: {}".format(weights))
    # norm_weights = [1 - (x / sum(class_sample_count)) for x in class_sample_count]
    # print("Norm weights: {}".format(norm_weights))
    samples_weight = torch.tensor([weights[int(t)] for t in target])
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight)) #replacement=False

    return sampler

def get_loaders(dataset_name, config, distributed=False):

    train_trans, val_trans = get_transformers(config['im_size'])

    train_ds = globals()[dataset_name](config['train_csv'], config, train_trans, train=True)
    val_ds = globals()[dataset_name](config['val_csv'], config, val_trans, train=False)

    sampler = get_weighted_sampler(train_ds.labels)
    if distributed:
        sampler = DistributedSamplerWrapper(sampler)
        print('Obtained Distributed Sampler')

    train_loader = DataLoader(dataset=train_ds, batch_size=config['batch_size'], sampler=sampler,
                              num_workers=config['num_workers'])  #, pin_memory=True
    val_loader = DataLoader(dataset=val_ds, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers']) #, pin_memory=True

    return train_loader, val_loader

class bloodmnist(torch.utils.data.Dataset):
    def __init__(self, csvfile, cfg, transform, train=False, use_cache=False):
        self.csvfile = csvfile
        self.cfg = cfg # config file
        self.transform = transform
        self.train = train

        self.augmix = self.cfg['augmix']
        self.datadir = self.cfg['data_dir']

        data = pd.read_csv(self.csvfile)
        self.indices = data['index'].values
        self.labels = data['labels'].values
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(self.labels)
        with open(os.path.join(self.cfg['data_dir'], 'bloodmnist_combined.pkl'), 'rb+') as f:
            tmp = pickle.load(f)
        f.close()
        self.images = tmp['images']

        self.use_cache = use_cache
        self.cache = dict()

    def __getitem__(self, idx):
        if idx in self.cache and self.use_cache:
            return self.cache[idx]

        img = self.images[self.indices[idx]]
        #img = img[...,::-1]   # Converting to BGR for consistency with other datasets
        #cropsize = int(img.shape[0]*1.0)
        #img = crop_center_numpy(img, cropsize) # img.shape = [h,w,c]
        #img = shade_of_gray_cc(img)

        if self.augmix and self.train:
            img = aug(Image.fromarray(img), self.transform, self.cfg['augmix_severity'])
        else:
            img = self.transform(Image.fromarray(img))

        label = np.array(self.labels[idx])
        label = torch.from_numpy(label)

        if self.use_cache:
            self.cache[idx] = (img, label)

        return img, label

    def __len__(self):
        return len(self.labels)

class tissuemnist():
    def __init__(self, csvfile, cfg, transform, train=False, use_cache=False):
        self.csvfile = csvfile
        self.cfg = cfg # config file
        self.augmix = self.cfg['augmix']
        self.datadir = self.cfg['data_dir']
        self.transform = transform
        self.train=train
        data = pd.read_csv(self.csvfile)
        self.indices = data['index'].values
        self.labels = data['labels'].values
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(self.labels)
        with open(os.path.join(self.cfg['data_dir'], 'tissuemnist_combined.pkl'), 'rb+') as f:
            tmp = pickle.load(f)
        f.close()
        self.images = tmp['images']

        self.use_cache = use_cache
        self.cache = dict()

    def __getitem__(self, idx):
        if idx in self.cache and self.use_cache:
            return self.cache[idx]

        img = self.images[self.indices[idx]]
        img = np.stack((img,)*3, axis=-1)  #Converting to RGB
        #cropsize = int(img.shape[0]*1.0)
        #img = crop_center_numpy(img, cropsize) # img.shape = [h,w,c]
        #img = shade_of_gray_cc(img)
        if self.augmix and self.train:
            img = aug(Image.fromarray(img), self.transform, self.cfg['augmix_severity'])
        else:
            img = self.transform(Image.fromarray(img))

        label = np.array(self.labels[idx])
        label = torch.from_numpy(label)

        if self.use_cache:
            self.cache[idx] = (img, label)

        return img, label

    def __len__(self):
        return len(self.labels)

class organcmnist(torch.utils.data.Dataset):
    def __init__(self, csvfile, cfg, transform, train=False, use_cache=False):
        self.csvfile = csvfile
        self.cfg = cfg # config file
        self.augmix = self.cfg['augmix']
        self.datadir = self.cfg['data_dir']
        self.transform = transform
        self.train=train
        data = pd.read_csv(self.csvfile)
        self.indices = data['index'].values
        self.labels = data['labels'].values
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(self.labels)
        with open(os.path.join(self.cfg['data_dir'], 'organcmnist_combined.pkl'), 'rb+') as f:
            tmp = pickle.load(f)
        f.close()
        self.images = tmp['images']

        self.use_cache = use_cache
        self.cache = dict()

    def __getitem__(self, idx):
        if idx in self.cache and self.use_cache:
            return self.cache[idx]

        img = self.images[self.indices[idx]]
        img = np.stack((img,)*3, axis=-1)  #Converting to RGB
        #cropsize = int(img.shape[0]*1.0)
        #img = crop_center_numpy(img, cropsize) # img.shape = [h,w,c]
        #img = shade_of_gray_cc(img)
        if self.augmix and self.train:
            img = aug(Image.fromarray(img), self.transform, self.cfg['augmix_severity'])
        else:
            img = self.transform(Image.fromarray(img))

        label = np.array(self.labels[idx])
        label = torch.from_numpy(label)

        if self.use_cache:
            self.cache[idx] = (img, label)

        return img, label

    def __len__(self):
        return len(self.labels)

class organamnist(torch.utils.data.Dataset):
    def __init__(self, csvfile, cfg, transform, train=False, use_cache=False):
        self.csvfile = csvfile
        self.cfg = cfg # config file
        self.augmix = self.cfg['augmix']
        self.datadir = self.cfg['data_dir']
        self.transform = transform
        self.train=train
        data = pd.read_csv(self.csvfile)
        self.indices = data['index'].values
        self.labels = data['labels'].values
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(self.labels)
        with open(os.path.join(self.cfg['data_dir'], 'organamnist_combined.pkl'), 'rb+') as f:
            tmp = pickle.load(f)
        f.close()
        self.images = tmp['images']

        self.use_cache = use_cache
        self.cache = dict()

    def __getitem__(self, idx):
        if idx in self.cache and self.use_cache:
            return self.cache[idx]

        img = self.images[self.indices[idx]]
        img = np.stack((img,)*3, axis=-1)  #Converting to RGB
        #cropsize = int(img.shape[0]*1.0)
        #img = crop_center_numpy(img, cropsize) # img.shape = [h,w,c]
        #img = shade_of_gray_cc(img)
        if self.augmix and self.train:
            img = aug(Image.fromarray(img), self.transform, self.cfg['augmix_severity'])
        else:
            img = self.transform(Image.fromarray(img))

        label = np.array(self.labels[idx])
        label = torch.from_numpy(label)

        if self.use_cache:
            self.cache[idx] = (img, label)

        return img, label

    def __len__(self):
        return len(self.labels)

class organsmnist(torch.utils.data.Dataset):
    def __init__(self, csvfile, cfg, transform, train=False, use_cache=False):
        self.csvfile = csvfile
        self.cfg = cfg # config file
        self.augmix = self.cfg['augmix']
        self.datadir = self.cfg['data_dir']
        self.transform = transform
        self.train=train
        data = pd.read_csv(self.csvfile)
        self.indices = data['index'].values
        self.labels = data['labels'].values
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(self.labels)
        with open(os.path.join(self.cfg['data_dir'], 'organsmnist_combined.pkl'), 'rb+') as f:
            tmp = pickle.load(f)
        f.close()
        self.images = tmp['images']

        self.use_cache = use_cache
        self.cache = dict()

    def __getitem__(self, idx):
        if idx in self.cache and self.use_cache:
            return self.cache[idx]

        img = self.images[self.indices[idx]]
        img = np.stack((img,)*3, axis=-1)  #Converting to RGB
        #cropsize = int(img.shape[0]*1.0)
        #img = crop_center_numpy(img, cropsize) # img.shape = [h,w,c]
        #img = shade_of_gray_cc(img)
        if self.augmix and self.train:
            img = aug(Image.fromarray(img), self.transform, self.cfg['augmix_severity'])
        else:
            img = self.transform(Image.fromarray(img))

        label = np.array(self.labels[idx])
        label = torch.from_numpy(label)

        if self.use_cache:
            self.cache[idx] = (img, label)

        return img, label

    def __len__(self):
        return len(self.labels)

class pathmnist(torch.utils.data.Dataset):
    def __init__(self, csvfile, cfg, transform, train=False, use_cache=False):
        self.csvfile = csvfile
        self.cfg = cfg # config file
        self.transform = transform
        self.train = train

        self.augmix = self.cfg['augmix']
        self.datadir = self.cfg['data_dir']

        data = pd.read_csv(self.csvfile)
        self.indices = data['index'].values
        self.labels = data['labels'].values
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(self.labels)
        with open(os.path.join(self.cfg['data_dir'], 'pathmnist_combined.pkl'), 'rb+') as f:
            tmp = pickle.load(f)
        f.close()
        self.images = tmp['images']

        self.use_cache = use_cache
        self.cache = dict()

    def __getitem__(self, idx):
        if idx in self.cache and self.use_cache:
            return self.cache[idx]

        img = self.images[self.indices[idx]].copy()
        img = img[...,::-1]   # Converting to BGR for consistency with other datasets
        #cropsize = int(img.shape[0]*1.0)
        #img = crop_center_numpy(img, cropsize) # img.shape = [h,w,c]
        #img = shade_of_gray_cc(img)

        if self.augmix and self.train:
            img = aug(Image.fromarray(img), self.transform, self.cfg['augmix_severity'])
        else:
            img = self.transform(Image.fromarray(img))

        label = np.array(self.labels[idx])
        label = torch.from_numpy(label)

        if self.use_cache:
            self.cache[idx] = (img, label)

        return img, label

    def __len__(self):
        return len(self.labels)

class dermamnist(torch.utils.data.Dataset):
    def __init__(self, csvfile, cfg, transform, train=False, use_cache=False):
        self.csvfile = csvfile
        self.cfg = cfg # config file
        self.transform = transform
        self.train = train

        self.augmix = self.cfg['augmix']
        self.datadir = self.cfg['data_dir']

        data = pd.read_csv(self.csvfile)
        self.indices = data['index'].values
        self.labels = data['labels'].values
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(self.labels)
        with open(os.path.join(self.cfg['data_dir'], 'dermamnist_combined.pkl'), 'rb+') as f:
            tmp = pickle.load(f)
        f.close()
        self.images = tmp['images']

        self.use_cache = use_cache
        self.cache = dict()

    def __getitem__(self, idx):
        if idx in self.cache and self.use_cache:
            return self.cache[idx]

        img = self.images[self.indices[idx]]
        #img = img[...,::-1]   # Converting to BGR for consistency with other datasets
        #cropsize = int(img.shape[0]*1.0)
        #img = crop_center_numpy(img, cropsize) # img.shape = [h,w,c]
        #img = shade_of_gray_cc(img)

        if self.augmix and self.train:
            img = aug(Image.fromarray(img), self.transform, self.cfg['augmix_severity'])
        else:
            img = self.transform(Image.fromarray(img))

        label = np.array(self.labels[idx])
        label = torch.from_numpy(label)

        if self.use_cache:
            self.cache[idx] = (img, label)

        return img, label

    def __len__(self):
        return len(self.labels)

class octmnist(torch.utils.data.Dataset):
    def __init__(self, csvfile, cfg, transform, train=False, use_cache=False):
        self.csvfile = csvfile
        self.cfg = cfg # config file
        self.augmix = self.cfg['augmix']
        self.datadir = self.cfg['data_dir']
        self.transform = transform
        self.train=train
        data = pd.read_csv(self.csvfile)
        self.indices = data['index'].values
        self.labels = data['labels'].values
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(self.labels)
        with open(os.path.join(self.cfg['data_dir'], 'octmnist_combined.pkl'), 'rb+') as f:
            tmp = pickle.load(f)
        f.close()
        self.images = tmp['images']

        self.use_cache = use_cache
        self.cache = dict()

    def __getitem__(self, idx):
        if idx in self.cache and self.use_cache:
            return self.cache[idx]

        img = self.images[self.indices[idx]]
        img = np.stack((img,)*3, axis=-1)  #Converting to RGB
        #cropsize = int(img.shape[0]*1.0)
        #img = crop_center_numpy(img, cropsize) # img.shape = [h,w,c]
        #img = shade_of_gray_cc(img)
        if self.augmix and self.train:
            img = aug(Image.fromarray(img), self.transform, self.cfg['augmix_severity'])
        else:
            img = self.transform(Image.fromarray(img))

        label = np.array(self.labels[idx])
        label = torch.from_numpy(label)

        if self.use_cache:
            self.cache[idx] = (img, label)

        return img, label

    def __len__(self):
        return len(self.labels)

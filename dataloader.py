import numpy as np
import os
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import Compose

from transforms import NormalizeIntensityd, RepeatChanneld, Resized, RandAffined,ScaleTo01


class Getfile(Dataset):
    def __init__(self, base_dir, image_dirs=None, domain=None, val_dir=None, num_classes=None, label_intensities=None,
                 mode=None, onehot=None, num_data=None, aug=False, vision=False):
        self._base_dir = base_dir
        self.image_dirs = image_dirs
        self.domain = domain
        self.val_dir = val_dir
        self.num_classes = num_classes
        self.label_intensities = label_intensities
        self.mode = mode
        self.onehot = onehot
        self.aug = aug
        self.vision = vision
        self.num_data = num_data
        self.path_pairs = []

        if self.val_dir is not None:
            self.image_list = os.listdir(os.path.join(self._base_dir, self.val_dir))
            print('load val image', len(self.image_list))
        else:
            self.s_train_dir = self.image_dirs["s_train_dir"][self.domain]
            self.s2t_train_dir = self.image_dirs["s2t_train_dir"][self.domain]
            self.image_list = os.listdir(os.path.join(self._base_dir, self.s_train_dir))
            self._load_csv_data()
        self.visual_transform = Compose([
            RandAffined(keys=['s', 's2t', 'label'], prob=0.5, rotate_range=45,
                        allow_missing_keys=True)
        ])
        self.pre_transform = Compose([
            ScaleTo01(keys=['s', 's2t']),
        ])
        self.transform = Compose([
        ])
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.normalize = Compose([
            Resized(keys=['s', 's2t', 'label'], spatial_size=(256, 256),
                    allow_missing_keys=True),
            RepeatChanneld(keys=['s', 's2t'], repeats=3, allow_missing_keys=True),
            NormalizeIntensityd(keys=['s', 's2t'], subtrahend=mean, divisor=std,
                                allow_missing_keys=True),
        ])
        print('Loading data from {}...'.format(self.mode))

    def _load_csv_data(self):
        csv_path = os.path.join(self._base_dir, self.s2t_train_dir, self.mode + '.csv')
        self.csv_data = pd.read_csv(csv_path)
        self.path_pairs = [
            (row['s_path'], row['s2t_paths']) for idx, row in self.csv_data.iterrows()
        ]

    def get_first_name(self, file_name):
        if 'fake' not in file_name:
            return file_name
        return file_name.split('_fake')[0] + '.npz'

    def create_csv_with_matching_files(self, output_csv_path):
        global s2t_paths
        data = []

        for s_file in self.s_train_list:
            first_name = self.get_first_name(s_file)
            s_path = os.path.join(self._base_dir, self.s_train_dir, s_file)

            for f in self.s2t_train_list:
                if self.get_first_name(f) == first_name:
                    s2t_paths = os.path.join(self._base_dir, self.s2t_train_dir, self.direct, f)

            data.append([s_path, s2t_paths])

        df = pd.DataFrame(data, columns=['s_path', 's2t_paths'])
        df.to_csv(output_csv_path, index=False)

    def _load_npz(self, image_path, label_path):
        image_data = np.load(image_path)
        data_vol = torch.from_numpy(image_data['image.npy'].astype(np.float32)).unsqueeze(0).float()

        if label_path is not None:
            label_data = np.load(label_path)
            label_vol = torch.from_numpy(label_data['label.npy'].astype(np.float32)).unsqueeze(0).float()
            return data_vol, label_vol

        return data_vol

    def __len__(self):
        if self.num_data == 0 or self.num_data is None:
            return len(self.image_list)
        return self.num_data

    def __getitem__(self, idx):
        if self.val_dir is not None:
            if self.num_data is None or self.num_data == 0:
                idx = idx % len(self.image_list)
            elif idx >= len(self.image_list):
                idx = random.randint(0, len(self.image_list) - 1)
            image_path = os.path.join(self._base_dir, self.val_dir, self.image_list[idx])
            image, label = self._load_npz(image_path, image_path)
            sample = {'s': image, 'label': label}
            sample = self.pre_transform(sample)
            sample = self.normalize(sample)

            if self.onehot:
                sample['label'] = get_one_hot_label(sample['label'], self.num_classes,
                                                    label_intensities=self.label_intensities)
            return sample
        else:
            if self.num_data is None or self.num_data == 0:
                idx = idx % len(self.path_pairs)
            elif idx >= len(self.path_pairs):
                idx = random.randint(0, len(self.path_pairs) - 1)

            s_path, s2t_path = self.path_pairs[idx]
            s_image, label = self._load_npz(s_path, s_path)
            s2t_image = self._load_npz(s2t_path, None)

            sample = {'s': s_image, 's2t': s2t_image, 'label': label}
            sample = self.pre_transform(sample)
            if self.aug:
                sample = self.transform(sample)
            if self.vision:
                sample = self.visual_transform(sample)
            sample = self.normalize(sample)

            if self.onehot:
                sample['label'] = get_one_hot_label(sample['label'], self.num_classes,
                                                    label_intensities=self.label_intensities)
            assert not torch.isnan(sample['s']).any(), "NaN found in source image"
            assert not torch.isinf(sample['s']).any(), "Inf found in source image"
            if 'label' in sample and sample['label'] is not None:
                assert torch.all((sample['label'].sum(dim=0) - 1.0).abs() < 1e-6) or torch.all(
                    sample['label'].sum(dim=0) == 0)
            return sample

    def get_filename(self, idx):
        return self.image_list[idx]


class MixedDataLoader(Sampler):
    def __init__(self, supervised_dataloader, unsupervised_dataloader, supervised_ratio=0.5):
        self.supervised_dataloader = supervised_dataloader
        self.unsupervised_dataloader = unsupervised_dataloader
        self.supervised_ratio = supervised_ratio

    def __len__(self):
        return len(self.unsupervised_dataloader)

    def __iter__(self):
        supervised_iter = iter(self.supervised_dataloader)
        unsupervised_iter = iter(self.unsupervised_dataloader)

        while True:
            try:
                unsupervised_batch = next(unsupervised_iter)
            except StopIteration:
                break

            if self.supervised_ratio > 0.0:
                try:
                    supervised_batch = next(supervised_iter)
                except StopIteration:
                    supervised_iter = iter(self.supervised_dataloader)
                    supervised_batch = next(supervised_iter)
            else:
                supervised_batch = None

            yield supervised_batch, unsupervised_batch


def get_one_hot_label(gt, num_classes, label_intensities=None, new_channel=False):
    if label_intensities is None:
        label_intensities = sorted(torch.unique(gt))
    label = torch.round(gt).to(torch.long)
    if new_channel:
        label = torch.zeros((num_classes, *label.shape), dtype=torch.float32)
        for k in range(num_classes):
            label[k] = (gt == label_intensities[k])
        label[0] = ~torch.sum(label[1:], dim=0).bool()

    else:
        label = torch.zeros((num_classes, *label.shape[1:]), dtype=torch.float32)
        for k in range(num_classes):
            label[k] = (gt == label_intensities[k])
        label[0] = ~torch.sum(label[1:], dim=0).bool()

    return label

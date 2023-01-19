from typing import Tuple, Union

import pathlib
import numpy as np
import torch
import torchvision
import yacs.config

from torch.utils.data import Dataset

from transforms import create_transform


class SubsetDataset(Dataset):
    def __init__(self, subset_dataset, transform=None):
        self.subset_dataset = subset_dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset_dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset_dataset)


def create_dataset(config: yacs.config.CfgNode,
                   is_train: bool) -> Union[Tuple[Dataset, Dataset], Dataset]:
    if config.dataset.name in [
            'CIFAR10',
            'CIFAR100',
            'MNIST',
            'FashionMNIST'
    ]:
        module = getattr(torchvision.datasets, config.dataset.name)
        if is_train:
            if config.train.use_test_as_val:
                train_transform = create_transform(config, is_train=True)
                val_transform = create_transform(config, is_train=False)
                train_dataset = module(config.dataset.dataset_dir,
                                       train=is_train,
                                       transform=train_transform,
                                       download=True)
                test_dataset = module(config.dataset.dataset_dir,
                                      train=False,
                                      transform=val_transform,
                                      download=True)
                # return train_dataset, test_dataset
            else:
                dataset = module(config.dataset.dataset_dir,
                                 train=is_train,
                                 transform=None,
                                 download=True)
                val_ratio = config.train.val_ratio
                assert val_ratio < 1
                val_num = int(len(dataset) * val_ratio)
                train_num = len(dataset) - val_num
                lengths = [train_num, val_num]
                train_subset, val_subset = torch.utils.data.dataset.random_split(
                    dataset, lengths)

                train_transform = create_transform(config, is_train=True)
                val_transform = create_transform(config, is_train=False)
                train_dataset = SubsetDataset(train_subset, train_transform)
                val_dataset = SubsetDataset(val_subset, val_transform)
                # return train_dataset, val_dataset
            if config.train.select == 'random':
                fisrt_num = int(len(train_dataset) * config.select.random)
                second_num = len(train_dataset) - fisrt_num
                lengths = [fisrt_num, second_num]
                train_dataset, _ = torch.utils.data.dataset.random_split(
                    train_dataset, lengths)
            elif config.train.select == 'byclass':
                if config.train.select.class_list == []:
                    target_classes = np.random.choice(
                        list(range(config.dataset.n_classes)),
                        config.select.n_class,
                        replace=False)
                else:
                    target_classes = config.train.select
                indices = list()
                for i in range(len(train_dataset)):
                    if train_dataset.targets[i] in target_classes:
                        indices.append(i)
                # Reassign train data and labels
                train_indx = np.array(indices)
                train_dataset.data = train_dataset.data[train_indx, :, :, :]
                train_dataset.targets = np.array(train_dataset.targets)[train_indx].tolist()
                indices = list()
                for j in range(len(test_dataset)):
                    if test_dataset.targets[j] in target_cls:
                        indices.append(j)
                test_indx = np.array(indices)
                test_dataset.data = test_dataset.data[test_indx, :, :, :]
                test_dataset.targets = np.array(test_dataset.targets)[test_indx].tolist()
                    

        else:
            transform = create_transform(config, is_train=False)
            dataset = module(config.dataset.dataset_dir,
                             train=is_train,
                             transform=transform,
                             download=True)
            return dataset
    # elif config.dataset.name == 'ImageNet':
    #     dataset_dir = pathlib.Path(config.dataset.dataset_dir).expanduser()
    #     train_transform = create_transform(config, is_train=True)
    #     val_transform = create_transform(config, is_train=False)
    #     train_dataset = torchvision.datasets.ImageFolder(
    #         dataset_dir / 'train', transform=train_transform)
    #     val_dataset = torchvision.datasets.ImageFolder(dataset_dir / 'val',
    #                                                    transform=val_transform)
    #     return train_dataset, val_dataset
    else:
        raise ValueError()

def select_by_class(dataset):
    '''
    select data samples by their labeled classes
    '''


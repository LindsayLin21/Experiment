from typing import Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import yacs.config

from torch.utils.data import DataLoader

from .datasets import create_dataset


def worker_init_fn(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def create_dataloader(config, is_train):
    if is_train:
        train_dataset, val_dataset = create_dataset(config, is_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            num_workers=config.train.dataloader.num_workers)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.validation.batch_size
            num_workers=config.validation.dataloader.num_workers)

        return train_loader, val_loader
    else:
        dataset = create_dataset(config, is_train)
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.test.batch_size,
            num_workers=config.test.dataloader.num_workers,
            shuffle=False)
        return test_loader
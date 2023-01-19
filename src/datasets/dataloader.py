from typing import Tuple, Union

import numpy as np
import torch
import yacs.config

from .datasets import create_dataset

def create_dataloader(config, is_train):
    if is_train:
        train_dataset, val_dataset = create_dataset(config, is_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            num_workers=config.train.dataloader.num_workers,
            shuffle=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.validation.batch_size,
            num_workers=config.validation.dataloader.num_workers,
            shuffle=False)
        return train_loader, val_loader
    else:
        dataset = create_dataset(config, is_train)
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.test.batch_size,
            num_workers=config.test.dataloader.num_workers,
            shuffle=False)
        return test_loader
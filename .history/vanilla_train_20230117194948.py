#!/usr/bin/env python

import argparse
import pathlib
import time

try:
    import apex
except ImportError:
    pass
import numpy as np
import torch
import torch.nn as nn
import torchvision

from fvcore.common.checkpoint import Checkpointer

from src import (
    create_dataloader,
    create_model,
    create_optimizer,
    create_scheduler,
    get_default_config,
    update_config
)

global_step = 0

criterion = nn.CrossEntropyLoss().cuda()
criterion.__init__(reduce=False)

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    if args.config is not None:
        config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    if not torch.cuda.is_available():
        config.device = 'cpu'
    if args.resume != '':
        config_path = pathlib.Path(args.resume) / 'config.yaml'
        config.merge_from_file(config_path.as_posix())
        config.merge_from_list(['train.resume', True])
    config = update_config(config)
    config.freeze()
    return config

def train(epoch, config, model, train_loader, optimizer, scheduler, logger):
    global global_step

    # logger.info(f'Train {epoch} {global_step}')

    device = torch.device(config.device)

    model.train()

    start_time = time.time()

    for step, (data, targets) in enumerate(train_loader, 0):
        step += 1
        global_step += 1

        data, targets = data.to(device), targets.to(device)

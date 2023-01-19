#!/usr/bin/env python

import argparse
import pathlib
import time
import sys

# try:
#     import apex
# except ImportError:
#     pass
import numpy as np
import torch
import torch.nn as nn
import torchvision

# from fvcore.common.checkpoint import Checkpointer

from src import (
    create_dataloader,
    create_model,
    create_initializer,
    create_optimizer,
    create_scheduler,
    get_default_config,
    update_config,
)

from src.utils import load_checkpoint, save_config

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

    # logger.info(f'Train {epoch} {global_step}')

    device = torch.device(config.device)

    model.train()

    for step, data in enumerate(train_loader, 0):
        step += 1

        data, targets = data.to(device), targets.to(device)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward propagation, compute loss, get predictions        
        _, outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        _, predicted = torch.max(outputs.data, 1)

        # Update loss, backward propagate, update optimizer
        loss = loss.mean()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        sys.stdout.write('\r')
        sys.stdout.write(
            '| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %
            (epoch+1, config.scheduler.epochs, step, len(train_loader), loss.item(),
            100. * correct / total)
        )
        sys.stdout.flush()

def main():
    config = load_config()

    set_seed(config)
    setup_cudnn(config)
    
    device = torch.device(config.device)

    # load data
    train_dataloader, valid_dataloader = create_dataloader(config, is_train=True)

    model = create_model(config)
    # model initiazalition
    init_mode = config.model.init_mode
    if init_mode == 'regress':
        old_model = load_checkpoint(config.regress.oldModel, device)
        model.apply(create_initializer(init_mode, old_model=old_model))
    elif init_mode in ['kaiming_fan_out', 'kaiming_fan_in']:
        model.apply(create_initializer(init_mode))

    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(optimizer, config)

    best_acc = 0.
    elapsed_time = 0




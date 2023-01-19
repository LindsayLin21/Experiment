#!/usr/bin/env python
import os
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

from src.utils import (
    get_hms,
    load_checkpoint, 
    save_config,
    set_seed,
    setup_cudnn,
    find_config_diff,
    get_env_info
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


def train(epoch, config, model, train_loader, optimizer):
    total = 0
    correct = 0
    train_loss = 0.

    device = torch.device(config.device)

    model.train()

    for step, data in enumerate(train_loader, 0):
        step += 1

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
            (epoch+1, config.train.epochs, step, len(train_loader), loss.item(),
            100. * correct / total)
        )
        sys.stdout.flush()

def validate(epoch, config, model, valid_loader, optimizer, best_acc=0):
    correct = 0
    total = 0
    test_loss = 0

    device = torch.device(config.device)

    model.eval()

    with torch.no_grad():
        for data in valid_loader:
            nputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            _, outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss = loss.mean()
            test_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100. * correct / total

    print("\n| Test Epoch #%d\t\t\tLoss: %.4f Acc: %.2f%% " %
            (epoch, loss.item(), test_acc))

    if test_acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' % (test_acc))
        state = {
            'epoch': epoch + 1, 
            'state_dict': model.state_dict(), 
            'optimizer': optimizer.state_dict()
            }
        torch.save(state, os.path.join(config.train.output_dir, f'checkpoint_{config.train.seed}'))
        best_acc = test_acc

    return best_acc


def main():
    config = load_config()

    set_seed(config)
    setup_cudnn(config)

    # epoch_seeds = np.random.randint(np.iinfo(np.int32).max // 2,
                                    # size=config.scheduler.epochs)

    output_dir = pathlib.Path(config.train.output_dir)
    output_dir = os.path.join(os.getcwd(), output_dir)
    if not config.train.resume and output_dir.exists():
        raise RuntimeError(
            f'Output directory `{output_dir.as_posix()}` already exists')
    output_dir.mkdir(exist_ok=True, parents=True)
    if not config.train.resume:
        save_config(config, output_dir / 'config.yaml')
        save_config(get_env_info(config), output_dir / 'env.yaml')
        diff = find_config_diff(config)
        if diff is not None:
            save_config(diff, output_dir / 'config_update.yaml')


    device = torch.device(config.device)

    # load data
    train_dataloader, valid_dataloader = create_dataloader(config, is_train=True)

    # resume from half-trained model
    if config.train.resume:
        if config.train.checkpoint != '':
            checkpoint = torch.load(config.train.checkpoint, map_location='cpu')
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
        else:
            raise FileNotFoundError('Path to checkpoint is absent')

    # model load and initiazalition
    model = create_model(config)
    init_mode = config.model.init_mode
    if init_mode == '':
        print('Default model initialization')
    elif init_mode == 'regress':
        if config.regress.oldModel == '':
            raise FileNotFoundError('Path to old model is absent')
        else:
            old_model = load_checkpoint(config.regress.oldModel, device)
            model.apply(create_initializer(init_mode, old_model=old_model))
    elif init_mode in ['kaiming_fan_out', 'kaiming_fan_in']:
        model.apply(create_initializer(init_mode))
    else:
        raise ValueError()

    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(optimizer, config)
    
    best_acc = 0.
    elapsed_time = 0

    for epoch in range(config.train.epochs):
        start_time = time.time()

        train(epoch, config, model, train_dataloader, optimizer)

        best_acc = validate(epoch, config, model, valid_dataloader, optimizer, best_acc=best_acc)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

        scheduler.step()

if __name__ == '__main__':
    main()





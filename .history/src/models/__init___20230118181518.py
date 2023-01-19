import importlib
from typing import Callable

import torch
import torch.nn as nn
import yacs.config


def create_model(config):
    module = importlib.import_module(
        'src.models'
        f'.{config.model.type}')
    model = getattr(module, f'.{config.model.name}')()
    device = torch.device(config.device)
    model.to(device)
    return model


def create_initializer(mode, old_model=None):
    if mode in ['kaiming_fan_out', 'kaiming_fan_in']:
        mode = mode[8:]

        def initializer(module):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data,
                                        mode=mode,
                                        nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight.data)
                nn.init.zeros_(module.bias.data)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data,
                                        mode=mode,
                                        nonlinearity='relu')
                nn.init.zeros_(module.bias.data)
    elif mode == 'regress':
        def initializer(model):
            old_model_params = {n:p for n, p in old_model.named_parameters() if p.requires_grad}
            for n, p in model.named_parameters():
                if n in old_model_params.keys():
                    old_weight = old_model_params[n]
                    if old_weight.shape == p.shape: # in case the fc layer is different
                        p.data = old_weight
    else:
        raise ValueError()

    return initializer

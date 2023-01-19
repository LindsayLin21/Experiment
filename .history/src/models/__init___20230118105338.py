import importlib

import torch
import torch.nn as nn
import yacs.config

from initializer import create_initializer

def create_model(config: yacs.config.CfgNode) -> nn.Module:
    module = importlib.import_module(
        'reduce_regress.models'
        f'.{config.model.type}.{config.model.name}')
    model = getattr(module, 'Network')(config)
    device = torch.device(config.device)
    model.to(device)
    return model
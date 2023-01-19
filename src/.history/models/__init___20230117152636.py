import importlib

import torch
import torch.nn as nn
import torch.distributed as dist
import yacs.config

def create_model(config: yacs.config.CfgNode) -> nn.Module:
    module = importlib.import_module(
        'pytorch_image_classification.models'
        f'.{config.model.type}.{config.model.name}')
    model = getattr(module, 'Network')(config)
    device = torch.device(config.device)
    model.to(device)
    return model
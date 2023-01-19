from typing import Callable

import torch.nn as nn


def create_initializer(mode: str) -> Callable:
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

    else:
        raise ValueError()

    return initializer

def custom_init_for_regress(model, original_model):
    '''
    Initialize new model weights with the original model (only for unchanged model architectures)
    '''
    old_model_params = {n:p for n, p in original_model.named_parameters() if p.requires_grad}

    for n, p in model.named_parameters():
        if n in old_model_params.keys():
            old_weight = old_model_params[n]
            if old_weight.shape == p.shape: # in case the fc layer is different
                p.data = old_weight
    # print('Initialization finished..')
    return model
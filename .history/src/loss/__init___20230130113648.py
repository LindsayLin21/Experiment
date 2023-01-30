import torch.nn as nn

def create_loss(config):
    loss_ce = nn.CrossEntropyLoss(reduction='mean')
    if config.regress.mode == '':
        return loss_ce
    elif config.regress.mode == 'nsr':
        loss_nsr = 0
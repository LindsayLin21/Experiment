import torch.nn as nn

from nsr_loss import NSRLoss

def create_loss(config):
    loss_ce = nn.CrossEntropyLoss(reduction='mean')
    if config.regress.mode == '':
        return loss_ce
    elif config.regress.mode == 'nsr':
        loss_regress = NSRLoss(config)
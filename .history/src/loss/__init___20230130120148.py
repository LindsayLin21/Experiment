import torch.nn as nn

from nsr_loss import NSRLoss

def create_loss():
    loss = nn.CrossEntropyLoss(reduction='mean')
    return loss

def create_regress_loss(config):
    if config.regress.mode == 'nsr':
        loss_regress = NSRLoss(config)
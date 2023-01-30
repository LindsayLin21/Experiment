import torch.nn as nn

from nsr_loss import NSRLoss
from qp_loss import QPLoss

def create_loss():
    loss = nn.CrossEntropyLoss(reduction='mean')
    return loss

def create_regress_loss(config):
    if config.regress.mode == 'nsr':
        loss = NSRLoss(config)
    elif config.regress.mode == 'qp':
        loss = QPLoss(config)

    return loss
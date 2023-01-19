import torch.nn as nn
def create_loss(config):
    loss = nn.CrossEntropyLoss(reduction='mean')
    return loss
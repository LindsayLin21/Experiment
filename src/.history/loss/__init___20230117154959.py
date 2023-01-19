import torch.nn as nn
def create_loss(config):
    train_loss = nn.CrossEntropyLoss(reduction='mean')
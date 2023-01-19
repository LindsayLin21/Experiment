from torch.optim.lr_scheduler import MultiStepLR

def create_scheduler(optimizer, config):
    lr_decay = config.scheduler.lr_decay
    scheduler = MultiStepLR(optimizer, milestones=config.scheduler.milestones, gamma=config.scheduler.lr_decay)

    return scheduler
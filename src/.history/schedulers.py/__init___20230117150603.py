import torch.optim.lr_scheduler

def create_scheduler(config, main_steps):
    lr_decay = config.scheduler.lr_decay
    scheduler = MultistepScheduler(main_steps, 1, lr_decay,
                                    config.scheduler.milestones)

    return scheduler
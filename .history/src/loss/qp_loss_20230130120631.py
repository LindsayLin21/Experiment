import torch

class QPLoss:
    def __init__(self, config) -> None:
        self.weight = config.regress.qp_weight
    
    def __call__(self, old_model, new_model) -> torch.Tensor:
        pass
    
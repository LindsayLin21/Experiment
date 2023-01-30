import torch

class QPLoss:
    def __init__(self, config) -> None:
        self.weight = config.regress.qp_weight
    
    def __call__(self, old_model, new_model) -> torch.Tensor:
        loss = 0.
        old_weights = {}

        old_weights = {n: p for n, p in old_model.named_parameters() if p.requires_grad}
        for n, p in new_model.named_parameters():
            if p.size() == old_weights[n].size(): # new and old model: the fully-connected layer could be in different dimension
                _loss = (p - old_weights[n]) ** 2 # generic quadratic penalty

        return loss * self.weight
    
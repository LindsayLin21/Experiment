from typing import Tuple

import torch
import torch.nn as nn
# from torch_scatter import scatter_sum

class NSRLoss:
    def __init__(self, config) -> None:
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.nsr_M = config.regress.nsr_M
        self.lb_queue = []
        self.y_queue = []
        self.weight = config.regress.nsr_weight

        self.device = torch.device(config.device)

    def __call__(
            self, response: torch.Tensor,
            targets: Tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
        
        response = response.view(response.size(0), -1) # reshape
        torch.cuda.empty_cache()

        if len(self.lb_queue) < self.nsr_M:
            self.y_queue.append(response.clone().detach())
            self.lb_queue.append(targets.clone().detach())
            return 0
        else:
            self.y_queue.pop(0)
            self.y_queue.append(response.clone().detach())
            self.lb_queue.pop(0)
            self.lb_queue.append(targets.clone().detach())

        # y_log_tensor = torch.cat((self.y_queue), 0).reshape(-1, response.size(1))
        # target_log_tensor = torch.cat((self.lb_queue), 0).reshape(-1, 1)
        # count_scatter = scatter_sum((y_log_tensor != 0).int(), target_log_tensor, dim=0).detach() # deal with the no response neurons, count minus 1
        # count_scatter = torch.where(count_scatter==0, torch.full_like(count_scatter, 1), count_scatter)
        # y_log_tensor_mean = scatter_sum(y_log_tensor, target_log_tensor, dim=0) / count_scatter
        # y_log_tensor_mean_scattered = torch.index_select(y_log_tensor_mean, 0, targets)
        # mask = torch.abs(response > 1e-5) # select the square greater than 1e-5
        # square_difference_the_class = (response - y_log_tensor_mean_scattered.to(self.device)) ** 2
        # square_difference_the_class = torch.masked_select(square_difference_the_class, mask)

        # return torch.mean(square_difference_the_class) * self.weight





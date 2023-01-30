from typing import Tuple

import torch
import torch.nn as nn

class NSRLoss:
    def __init__(self, config) -> None:
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.nsr_M = config.regress.nsr_M

    def __call__(
            self, responses: torch.Tensor,
            targets: Tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
        
        


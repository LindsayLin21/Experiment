from typing import Tuple

import torch
import torch.nn as nn

class NSRLoss:
    def __init__(self) -> None:
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def __call__(
            self, response: torch.Tensor,
            targets: Tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
        pass
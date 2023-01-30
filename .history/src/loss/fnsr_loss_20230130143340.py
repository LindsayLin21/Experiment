import torch

class FNSR_Loss:
    def __init__(self, config) -> None:
        self.device = torch.device(config.device)
        
        
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
import pathlib
import random

import numpy as np
import torch
import yacs.config

def set_seed(config: yacs.config.CfgNode) -> None:
    seed = config.train.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def setup_cudnn(config: yacs.config.CfgNode) -> None:
    torch.backends.cudnn.benchmark = config.cudnn.benchmark
    torch.backends.cudnn.deterministic = config.cudnn.deterministic

def save_config(config: yacs.config.CfgNode,
                output_path: pathlib.Path) -> None:
    with open(output_path, 'w') as f:
        f.write(str(config))

def load_checkpoint(path, device):
    model = torch.load(path, map_location=device)
    return model

def get_env_info(config: yacs.config.CfgNode) -> yacs.config.CfgNode:
    info = {
        'pytorch_version': str(torch.__version__),
        'cuda_version': torch.version.cuda or '',
        'cudnn_version': torch.backends.cudnn.version() or '',
    }
    if config.device != 'cpu':
        info['num_gpus'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        info['gpu_capability'] = f'{capability[0]}.{capability[1]}'

    return ConfigNode({'env_info': info})
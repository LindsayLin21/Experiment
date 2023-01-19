import pathlib
import random

import numpy as np
import torch
import yacs.config

from src.config.config_node import ConfigNode

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

def find_config_diff(
        config: yacs.config.CfgNode) -> Optional[yacs.config.CfgNode]:
    def _find_diff(node: yacs.config.CfgNode,
                   default_node: yacs.config.CfgNode):
        root_node = ConfigNode()
        for key in node:
            val = node[key]
            if isinstance(val, yacs.config.CfgNode):
                new_node = _find_diff(node[key], default_node[key])
                if new_node is not None:
                    root_node[key] = new_node
            else:
                if node[key] != default_node[key]:
                    root_node[key] = node[key]
        return root_node if len(root_node) > 0 else None

    default_config = get_default_config()
    new_config = _find_diff(config, default_config)
    return new_config
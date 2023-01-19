from .config_node import ConfigNode

config = ConfigNode()

config.device = 'cuda'
# cuDNN
config.cudnn = ConfigNode()
config.cudnn.benchmark = True
config.cudnn.deterministic = False

config.dataset = ConfigNode()
config.dataset.name = 'CIFAR10'
config.dataset.dataset_dir = ''
config.dataset.image_size = 32
config.dataset.n_channels = 3
config.dataset.n_classes = 10

config.model = ConfigNode()
# options: 'cifar', 'imagenet'
# Use 'cifar' for small input images
config.model.type = 'cifar'
config.model.name = 'resnet_preact'

config.model.resnet = ConfigNode()
config.model.resnet.depth = 110  # for cifar type model
config.model.resnet.n_blocks = [2, 2, 2, 2]  # for cifar type model
config.model.resnet.block_type = 'basic'
config.model.resnet.initial_channels = 16
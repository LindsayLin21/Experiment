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

config.model.densenet = ConfigNode()
config.model.densenet.depth = 100  # for cifar type model
config.model.densenet.n_blocks = [6, 12, 24, 16]  # for imagenet type model
config.model.densenet.block_type = 'bottleneck'
config.model.densenet.growth_rate = 12
config.model.densenet.drop_rate = 0.0
config.model.densenet.compression_rate = 0.5

config.model.vgg = ConfigNode()
config.model.vgg.n_channels = [64, 128, 256, 512, 512]
config.model.vgg.n_layers = [2, 2, 3, 3, 3]
config.model.vgg.use_bn = True

config.train = ConfigNode()
config.train.checkpoint = ''
config.train.resume = False
config.train.precision = 'O0'
config.train.batch_size = 64


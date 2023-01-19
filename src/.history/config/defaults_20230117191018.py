from .config_node import ConfigNode

config = ConfigNode()

config.device = 'cuda'
# cuDNN
config.cudnn = ConfigNode()
config.cudnn.benchmark = True
config.cudnn.deterministic = False

# original model
config.regress = False
config.regress.oldModel = ''
config.regress.old_nClasses = 10

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
config.train.use_test_as_val = False
config.train.val_ratio = 0.1 # percent of train/validation data (random) split

# select training data
config.train.selectmode = '' # random, byclass
config.train.selectmode.random = 0.5
config.train.selectmode.byclass = [] # select target classes in training data (random or decisive)

# optimizer (options: sgd, adam)
config.train.optimizer = 'sgd'
config.train.base_lr = 0.1
config.train.momentum = 0.9
config.train.nesterov = True
config.train.weight_decay = 1e-4
config.train.no_weight_decay_on_bn = False
config.train.start_epoch = 0
config.train.seed = 0

config.train.output_dir = '/exp00'
config.train.log_period = 100
config.train.checkpoint_period = 10

# optimizer
config.optim = ConfigNode()
# Adam
config.optim.adam = ConfigNode()
config.optim.adam.betas = (0.9, 0.999)

# scheduler
config.scheduler = ConfigNode()
config.scheduler.epochs = 200
# main scheduler (multistep)
config.scheduler.type = 'multistep'
config.scheduler.milestones = [80, 120]
config.scheduler.lr_decay = 0.1

# train data loader
config.train.dataloader = ConfigNode()
config.train.dataloader.num_workers = 2

# validation data loader
config.validation = ConfigNode()
config.validation.batch_size = 64
config.validation.dataloader = ConfigNode()
config.validation.dataloader.num_workers = 2

config.augmentation = ConfigNode()
config.augmentation.use_random_crop = False
config.augmentation.use_random_horizontal_flip = False
config.augmentation.use_cutout = False

config.augmentation.random_crop = ConfigNode()
config.augmentation.random_crop.padding = 4
config.augmentation.random_crop.fill = 0
config.augmentation.random_crop.padding_mode = 'constant'

config.augmentation.random_horizontal_flip = ConfigNode()
config.augmentation.random_horizontal_flip.prob = 0.5

config.augmentation.cutout = ConfigNode()
config.augmentation.cutout.n_holes = 1 # option for cifar type
config.augmentation.cutout.length = 16 # option for cifar type

# test config
config.test = ConfigNode()
config.test.checkpoint = ''
config.test.output_dir = ''
config.test.batch_size = 64
# test data loader
config.test.dataloader = ConfigNode()
config.test.dataloader.num_workers = 2
config.test.dataloader.pin_memory = False


def get_default_config():
    return config.clone()
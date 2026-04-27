from cifar_config import CIFAR10Config

class CIFAR100Config(CIFAR10Config):
    dataset = 'CIFAR100'
    num_classes = 100  # Only change
    epochs = 50
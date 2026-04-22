class CIFAR10Config:
    # Data
    dataset = 'CIFAR10'
    img_size = 32
    num_classes = 10
    batch_size = 128
    
    # Model - ViT-Small (larger than MNIST)
    patch_size = 4  # 32/4 = 8 patches per side = 64 total
    embed_dim = 384
    num_layers = 8
    num_heads = 6
    mlp_ratio = 4
    dropout = 0.1
    
    # Training
    epochs = 100
    lr = 0.0003
    weight_decay = 0.05
    warmup_epochs = 5
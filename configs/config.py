class Config:
    # Data
    dataset = 'MNIST'
    img_size = 28
    num_classes = 10
    batch_size = 128
    
    # Model (ViT-Tiny for MNIST)
    patch_size = 4  # 28/4 = 7 patches per side = 49 total
    embed_dim = 192
    num_layers = 6  # Smaller for MNIST
    num_heads = 4
    mlp_ratio = 4
    dropout = 0.1
    
    # Training
    epochs = 20
    lr = 0.001
    weight_decay = 0.0001
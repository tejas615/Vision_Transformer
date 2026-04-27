import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_training_curves(log_file, save_path):
    # Parse log file
    epochs, train_loss, train_acc, test_loss, test_acc = [], [], [], [], []
    
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
        
            # Skip empty lines or header
            if not line or line.startswith("epoch"):
                continue
        
            parts = line.split(',')
        
            epochs.append(int(parts[0]))
            train_loss.append(float(parts[1]))
            train_acc.append(float(parts[2]))
            test_loss.append(float(parts[3]))
            test_acc.append(float(parts[4]))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(epochs, train_loss, label='Train')
    ax1.plot(epochs, test_loss, label='Test')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, train_acc, label='Train')
    ax2.plot(epochs, test_acc, label='Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    


def visualize_attention(model, image, save_path):
    """Visualize attention maps from different layers"""
    model.eval()
    
    # Hook to capture attention weights
    attention_maps = []
    
    def hook_fn(module, input, output):
        # output[1] is attention weights from MultiheadAttention
        attention_maps.append(output[1].detach())
    
    # Register hooks on attention layers
    hooks = []
    for block in model.blocks:
        hook = block.attn.register_forward_hook(hook_fn)
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(image.unsqueeze(0))
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Plot attention maps
    num_layers = len(attention_maps)
    fig, axes = plt.subplots(2, num_layers//2, figsize=(15, 6))
    
    for i, attn in enumerate(attention_maps):
        # Average over heads, get CLS token attention to patches
        attn_map = attn[0].mean(0)[0, 1:].reshape(8, 8)  # 64 patches → 8×8
        
        ax = axes[i // (num_layers//2), i % (num_layers//2)]
        im = ax.imshow(attn_map.cpu(), cmap='viridis')
        ax.set_title(f'Layer {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
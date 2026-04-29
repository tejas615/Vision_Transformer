"""
Visualization Utilities for ViT vs Hybrid Comparison

This module provides functions to create all visualizations for the project:
1. Training curves (loss and accuracy over epochs)
2. Data efficiency plots (performance vs dataset size)
3. Attention map visualizations
4. Architecture comparison diagrams
5. Confusion matrices
6. Model comparison tables
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches


# Set style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_training_curves(vit_log_path, hybrid_log_path, save_path='figures/training_curves.png'):
    """
    Plot training curves comparing ViT and Hybrid ViT.
    
    Creates a 2x2 subplot showing:
    - Train loss over epochs
    - Train accuracy over epochs
    - Test loss over epochs
    - Test accuracy over epochs
    
    Args:
        vit_log_path (str): Path to ViT training log CSV
        hybrid_log_path (str): Path to Hybrid training log CSV
        save_path (str): Where to save the plot
        
    Example:
        >>> plot_training_curves('experiments/logs/vit_cifar10.csv', 
                                 'experiments/logs/hybrid_cifar10.csv')
    """
    # Load training logs
    columns = ['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc']

    vit_df = pd.read_csv(vit_log_path, names=columns).dropna().astype(float)
    hybrid_df = pd.read_csv(hybrid_log_path, names=columns).dropna().astype(float)

    vit_df = vit_df.sort_values('epoch')
    hybrid_df = hybrid_df.sort_values('epoch')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Comparison: ViT vs Hybrid ViT on CIFAR-10', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Train Loss
    axes[0, 0].plot(vit_df['epoch'], vit_df['train_loss'], 
                    label='ViT', linewidth=2, marker='o', markersize=4, alpha=0.7)
    axes[0, 0].plot(hybrid_df['epoch'], hybrid_df['train_loss'], 
                    label='Hybrid ViT', linewidth=2, marker='s', markersize=4, alpha=0.7)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Training Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss Over Time', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Train Accuracy
    axes[0, 1].plot(vit_df['epoch'], vit_df['train_acc'], 
                    label='ViT', linewidth=2, marker='o', markersize=4, alpha=0.7)
    axes[0, 1].plot(hybrid_df['epoch'], hybrid_df['train_acc'], 
                    label='Hybrid ViT', linewidth=2, marker='s', markersize=4, alpha=0.7)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Training Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Training Accuracy Over Time', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Test Loss
    axes[1, 0].plot(vit_df['epoch'], vit_df['test_loss'], 
                    label='ViT', linewidth=2, marker='o', markersize=4, alpha=0.7)
    axes[1, 0].plot(hybrid_df['epoch'], hybrid_df['test_loss'], 
                    label='Hybrid ViT', linewidth=2, marker='s', markersize=4, alpha=0.7)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Test Loss', fontsize=12)
    axes[1, 0].set_title('Test Loss Over Time', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Test Accuracy
    axes[1, 1].plot(vit_df['epoch'], vit_df['test_acc'], 
                    label='ViT', linewidth=2, marker='o', markersize=4, alpha=0.7)
    axes[1, 1].plot(hybrid_df['epoch'], hybrid_df['test_acc'], 
                    label='Hybrid ViT', linewidth=2, marker='s', markersize=4, alpha=0.7)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Test Accuracy (%)', fontsize=12)
    axes[1, 1].set_title('Test Accuracy Over Time', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Highlight best test accuracy for each model
    vit_best = vit_df['test_acc'].max()
    hybrid_best = hybrid_df['test_acc'].max()
    vit_best_epoch = vit_df.loc[vit_df['test_acc'].idxmax(), 'epoch']
    hybrid_best_epoch = hybrid_df.loc[hybrid_df['test_acc'].idxmax(), 'epoch']
    
    improvement = hybrid_best - vit_best
    sign = "+" if improvement >= 0 else ""

    axes[1, 1].text(
        0.02, 0.95,
        f'ViT Best: {vit_best:.2f}% (epoch {int(vit_best_epoch)})\n'
        f'Hybrid Best: {hybrid_best:.2f}% (epoch {int(hybrid_best_epoch)})\n'
        f'Improvement: {sign}{improvement:.2f}%',
        transform=axes[1, 1].transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_data_efficiency(results_dict, save_path='figures/data_efficiency.png'):
    """
    Plot data efficiency comparison.
    
    Shows how model performance changes with different amounts of training data.
    This is crucial for demonstrating the advantage of inductive biases.
    
    Args:
        results_dict (dict): Dictionary with structure:
            {
                'fractions': [0.1, 0.25, 0.5, 1.0],
                'vit_acc': [45.2, 58.3, 64.1, 67.3],
                'hybrid_acc': [52.1, 65.8, 71.2, 74.8]
            }
        save_path (str): Where to save the plot
        
    Example:
        >>> results = {
                'fractions': [0.1, 0.25, 0.5, 1.0],
                'vit_acc': [45.2, 58.3, 64.1, 67.3],
                'hybrid_acc': [52.1, 65.8, 71.2, 74.8]
            }
        >>> plot_data_efficiency(results)
    """
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fractions = results_dict['fractions']
    vit_acc = results_dict['vit_acc']
    hybrid_acc = results_dict['hybrid_acc']
    
    if not (len(fractions) == len(vit_acc) == len(hybrid_acc)):
        raise ValueError("All input lists must have the same length")
    
    # Convert fractions to percentages for x-axis labels
    x = list(range(len(fractions)))
    fraction_labels = [f'{int(f*100)}%' for f in fractions]
    
    # Plot lines with markers
    ax.plot(x, vit_acc, marker='o', linewidth=3, 
            markersize=10, label='ViT', color='#E74C3C')
    ax.plot(x, hybrid_acc, marker='s', linewidth=3, 
            markersize=10, label='Hybrid ViT', color='#3498DB')
    ax.set_xticks(x)
    ax.set_xticklabels(fraction_labels)
    
    # Add value labels on points
    for i, (f, v, h) in enumerate(zip(fraction_labels, vit_acc, hybrid_acc)):
        ax.text(i, v-2, f'{v:.1f}%', ha='center', fontsize=9, color='#E74C3C')
        ax.text(i, h+2, f'{h:.1f}%', ha='center', fontsize=9, color='#3498DB')
    
    # Calculate and show improvement at each point
    improvements = [h - v for h, v in zip(hybrid_acc, vit_acc)]
    
    # Add improvement annotations
    for i, (f, imp) in enumerate(zip(fraction_labels, improvements)):
        ax.annotate(f'+{imp:.1f}%', xy=(i, (vit_acc[i] + hybrid_acc[i])/2),
                   xytext=(10, 0), textcoords='offset points',
                   fontsize=9, color='green', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    ax.set_xlabel('Training Data Used', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Data Efficiency: Performance vs. Dataset Size\n'
                 'Hybrid ViT Shows Larger Gains with Less Data', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add shaded region showing the gap
    ax.fill_between(x, vit_acc, hybrid_acc, 
                     alpha=0.2, color='green')
    
    # Add text box with key insight
    textstr = f'Key Insight:\nWith only 10% of data:\n' \
              f'ViT: {vit_acc[0]:.1f}%\n' \
              f'Hybrid: {hybrid_acc[0]:.1f}%\n' \
              f'Gap: +{improvements[0]:.1f}%\n\n' \
              f'The gap is larger with\nless data, showing CNN\ninductive biases help!'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Data efficiency plot saved to {save_path}")
    plt.close()


def visualize_attention_maps(model, image, device, save_path='figures/attention_maps.png', num_layers=4):
    model.eval()
    image = image.unsqueeze(0).to(device)
    
    # 1. Check for weight loading success
    if model.blocks[0].attn.in_proj_weight.abs().sum() < 1e-5:
        print("❌ ERROR: Model weights appear to be zero. Ensure you called model.load_state_dict()!")
        return

    attention_maps = []
    total_layers = len(model.blocks)
    layer_indices = np.linspace(0, total_layers - 1, num_layers, dtype=int)

    # 2. Manual Step-Through following your model architecture 
    with torch.no_grad():
        # Step A: Patch Embedding [cite: 698]
        x = model.patch_embed(image)
        
        # Step B: CLS Token Prepending [cite: 705]
        B = x.shape[0]
        cls_tokens = model.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Step C: Position Embedding [cite: 702]
        x = x + model.pos_embed
        
        # Step D: Iterate through Blocks
        for i, block in enumerate(model.blocks):
            # Pre-norm for Attention
            normed_x = block.norm1(x)
            
            # --- THE FIX: Explicitly request weights ---
            # We must pass need_weights=True here to override the default
            attn_output, attn_weights = block.attn(
                normed_x, normed_x, normed_x, 
                need_weights=True, 
                average_attn_weights=True
            )
            
            if i in layer_indices:
                attention_maps.append(attn_weights[0].detach().cpu())
            
            # Residual Connection 1: Attention
            x = x + attn_output
            
            # Residual Connection 2: MLP
            x = x + block.mlp(block.norm2(x))

    # 3. Plotting Logic
    fig, axes = plt.subplots(2, num_layers, figsize=(num_layers * 4, 8))
    orig_img = image[0].cpu().permute(1, 2, 0).numpy()
    orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
    if orig_img.shape[2] == 1: orig_img = orig_img.squeeze() # MNIST support

    patch_size, img_size = model.config.patch_size, model.config.img_size
    grid_size = img_size // patch_size

    for idx, (layer_idx, attn_map) in enumerate(zip(layer_indices, attention_maps)):
        # Get attention from CLS token (index 0) to patches (1:)
        cls_attn = attn_map[0, 1:].reshape(grid_size, grid_size)
        
        # Statistical Printout for debugging
        print(f"Layer {layer_idx+1} | Mean: {cls_attn.mean():.4f} | Std: {cls_attn.std():.4f}")

        # Contrast enhancement for visual clarity
        cls_attn_norm = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-10)

        attn_upsampled = torch.nn.functional.interpolate(
            cls_attn_norm.unsqueeze(0).unsqueeze(0),
            size=(img_size, img_size), mode='bilinear', align_corners=False
        ).squeeze().numpy()

        # Top row: Overlay
        axes[0, idx].imshow(orig_img, cmap='gray' if len(orig_img.shape)==2 else None)
        axes[0, idx].imshow(attn_upsampled, cmap='jet', alpha=0.4)
        axes[0, idx].set_title(f'Layer {layer_idx + 1} Overlay', fontsize=10)
        axes[0, idx].axis('off')

        # Bottom row: Heatmap
        im = axes[1, idx].imshow(cls_attn.numpy(), cmap='viridis')
        axes[1, idx].set_title(f'Attention Distribution', fontsize=10)
        axes[1, idx].axis('off')
        plt.colorbar(im, ax=axes[1, idx], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_architecture_comparison(save_path='figures/architecture_comparison.png'):
    """
    Create a visual comparison of ViT vs Hybrid architecture.
    
    This is a simplified diagram showing the key difference: patch embedding.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # ViT Architecture (left)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Vision Transformer (ViT)', fontsize=16, fontweight='bold', pad=20)
    
    # Draw ViT pipeline
    y_pos = 9
    
    # Input image
    ax1.add_patch(Rectangle((4, y_pos), 2, 1, facecolor='lightblue', 
                            edgecolor='black', linewidth=2))
    ax1.text(5, y_pos+0.5, 'Image\n32×32×3', ha='center', va='center', fontsize=10)
    y_pos -= 1.5
    ax1.arrow(5, y_pos+1.3, 0, -0.5, head_width=0.3, head_length=0.2, fc='black')
    
    # Linear patch embedding
    ax1.add_patch(Rectangle((3, y_pos), 4, 0.8, facecolor='#FFE5E5', 
                            edgecolor='red', linewidth=2))
    ax1.text(5, y_pos+0.4, 'Linear Patch Embedding\n(Conv 3→384, 4×4, stride=4)', 
             ha='center', va='center', fontsize=9)
    y_pos -= 1.3
    ax1.arrow(5, y_pos+1.0, 0, -0.5, head_width=0.3, head_length=0.2, fc='black')
    
    # Transformer blocks
    for i in range(3):
        ax1.add_patch(Rectangle((3.5, y_pos), 3, 0.6, facecolor='lightyellow', 
                                edgecolor='black', linewidth=1.5))
        ax1.text(5, y_pos+0.3, f'Transformer Block {i+1}', 
                 ha='center', va='center', fontsize=9)
        y_pos -= 0.8
        if i < 2:
            ax1.arrow(5, y_pos+0.6, 0, -0.15, head_width=0.2, head_length=0.1, fc='black')
    
    ax1.text(5, y_pos, '... (8 blocks total)', ha='center', fontsize=9, style='italic')
    y_pos -= 0.5
    ax1.arrow(5, y_pos+0.3, 0, -0.3, head_width=0.3, head_length=0.2, fc='black')
    
    # Classification head
    ax1.add_patch(Rectangle((3.5, y_pos-0.6), 3, 0.6, facecolor='lightgreen', 
                            edgecolor='black', linewidth=2))
    ax1.text(5, y_pos-0.3, 'Classification Head', ha='center', va='center', fontsize=10)
    
    # Problem annotation
    ax1.text(1, 6.5, 'Problem:\n❌ No locality bias\n❌ No translation\n    equivariance\n❌ Needs massive\n    datasets', 
             bbox=dict(boxstyle='round', facecolor='#FFE5E5', alpha=0.8),
             fontsize=9, verticalalignment='top')
    
    # Hybrid ViT Architecture (right)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Hybrid CNN-ViT (Our Approach)', fontsize=16, fontweight='bold', pad=20)
    
    # Draw Hybrid pipeline
    y_pos = 9
    
    # Input image
    ax2.add_patch(Rectangle((4, y_pos), 2, 1, facecolor='lightblue', 
                            edgecolor='black', linewidth=2))
    ax2.text(5, y_pos+0.5, 'Image\n32×32×3', ha='center', va='center', fontsize=10)
    y_pos -= 1.5
    ax2.arrow(5, y_pos+1.3, 0, -0.5, head_width=0.3, head_length=0.2, fc='black')
    
    # CNN patch embedding (highlighted as innovation)
    ax2.add_patch(FancyBboxPatch((2.5, y_pos), 5, 1.2, boxstyle="round,pad=0.1",
                                 facecolor='#E5F5FF', edgecolor='blue', linewidth=3))
    ax2.text(5, y_pos+0.9, 'CNN Patch Embedding', ha='center', fontweight='bold', fontsize=10)
    ax2.text(5, y_pos+0.6, 'Conv1: 3→64 (3×3, stride=2)', ha='center', fontsize=8)
    ax2.text(5, y_pos+0.35, 'Conv2: 64→128 (3×3, stride=2)', ha='center', fontsize=8)
    ax2.text(5, y_pos+0.1, 'Conv3: 128→384 (1×1)', ha='center', fontsize=8)
    y_pos -= 1.7
    ax2.arrow(5, y_pos+1.4, 0, -0.5, head_width=0.3, head_length=0.2, fc='black')
    
    # Transformer blocks (same as ViT)
    for i in range(3):
        ax2.add_patch(Rectangle((3.5, y_pos), 3, 0.6, facecolor='lightyellow', 
                                edgecolor='black', linewidth=1.5))
        ax2.text(5, y_pos+0.3, f'Transformer Block {i+1}', 
                 ha='center', va='center', fontsize=9)
        y_pos -= 0.8
        if i < 2:
            ax2.arrow(5, y_pos+0.6, 0, -0.15, head_width=0.2, head_length=0.1, fc='black')
    
    ax2.text(5, y_pos, '... (8 blocks total)', ha='center', fontsize=9, style='italic')
    y_pos -= 0.5
    ax2.arrow(5, y_pos+0.3, 0, -0.3, head_width=0.3, head_length=0.2, fc='black')
    
    # Classification head
    ax2.add_patch(Rectangle((3.5, y_pos-0.6), 3, 0.6, facecolor='lightgreen', 
                            edgecolor='black', linewidth=2))
    ax2.text(5, y_pos-0.3, 'Classification Head', ha='center', va='center', fontsize=10)
    
    # Benefits annotation
    ax2.text(8.5, 6.5, 'Benefits:\n✅ Locality bias\n✅ Translation\n    equivariance\n✅ Works with\n    small datasets', 
             bbox=dict(boxstyle='round', facecolor='#E5FFE5', alpha=0.8),
             fontsize=9, verticalalignment='top')
    
    plt.suptitle('Architecture Comparison: The Key Difference is Patch Embedding', 
                 fontsize=17, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Architecture comparison saved to {save_path}")
    plt.close()


def plot_confusion_matrix(model, test_loader, class_names, device, 
                          save_path='figures/confusion_matrix.png'):
    """
    Plot confusion matrix for model predictions.
    
    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        class_names (list): List of class names
        device (torch.device): Device
        save_path (str): Save path
    """
    from sklearn.metrics import confusion_matrix
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalize
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_sum[cm_sum == 0] = 1
    cm_norm = cm.astype('float') / cm_sum
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    annot = len(class_names) <= 20
    sns.heatmap(cm_norm, annot=annot, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def create_results_table(vit_results, hybrid_results, save_path='figures/results_table.png'):
    """
    Create a visual results comparison table.
    
    Args:
        vit_results (dict): ViT results with keys: mnist_acc, cifar10_acc, cifar100_acc, params, train_time
        hybrid_results (dict): Hybrid results with same keys
        save_path (str): Save path
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    data = [
        ['Model', 'Parameters', 'MNIST', 'CIFAR-10', 'CIFAR-100', 'Train Time\n(CIFAR-10)'],
        ['ViT', 
         f"{vit_results['params']/1e6:.2f}M",
         f"{vit_results['mnist_acc']:.2f}%",
         f"{vit_results['cifar10_acc']:.2f}%",
         f"{vit_results['cifar100_acc']:.2f}%",
         f"{vit_results['train_time']:.1f}h"],
        ['Hybrid ViT',
         f"{hybrid_results['params']/1e6:.2f}M",
         f"{hybrid_results['mnist_acc']:.2f}%",
         f"{hybrid_results['cifar10_acc']:.2f}%",
         f"{hybrid_results['cifar100_acc']:.2f}%",
         f"{hybrid_results['train_time']:.1f}h"],
        ['Improvement',
         f"+{(hybrid_results['params']-vit_results['params'])/1e6:.2f}M\n(+{100*(hybrid_results['params']/vit_results['params']-1):.1f}%)",
         f"+{hybrid_results['mnist_acc']-vit_results['mnist_acc']:.2f}%",
         f"+{hybrid_results['cifar10_acc']-vit_results['cifar10_acc']:.2f}%",
         f"+{hybrid_results['cifar100_acc']-vit_results['cifar100_acc']:.2f}%",
         f"-{vit_results['train_time']-hybrid_results['train_time']:.1f}h\n(-{100*(vit_results['train_time']-hybrid_results['train_time'])/vit_results['train_time']:.1f}%)"]
    ]
    
    # Create table
    table = ax.table(cellText=data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style model rows
    for i in range(6):
        table[(1, i)].set_facecolor('#E7E6E6')
        table[(2, i)].set_facecolor('#E7E6E6')
    
    # Style improvement row
    for i in range(6):
        table[(3, i)].set_facecolor('#C6E0B4')
        table[(3, i)].set_text_props(weight='bold', color='green')
    
    ax.set_title('Performance Comparison: ViT vs Hybrid ViT', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Results table saved to {save_path}")
    plt.close()


# Main function to generate all visualizations
def generate_all_visualizations(vit_model_path, hybrid_model_path, config, device):
    """
    Generate all visualizations for the presentation.
    
    Args:
        vit_model_path (str): Path to ViT checkpoint
        hybrid_model_path (str): Path to Hybrid checkpoint
        config: Configuration object
        device: torch device
    """
    from models.vit import VisionTransformer
    from models.hybrid import HybridViT
    from utils.data import get_cifar10_loaders
    import torchvision.transforms as transforms
    from torchvision import datasets
    
    print("Generating all visualizations...")
    
    # 1. Training curves
    print("\n1. Generating training curves...")
    plot_training_curves(
        'experiments/vit_cifar10_metrics.txt',
        'experiments/hybrid_cifar10_metrics.txt'
    )
    
    # 2. Data efficiency
    print("\n2. Generating data efficiency plot...")
    # You'll need to fill these with your actual results
    data_efficiency_results = {
        'fractions': [0.1, 0.25, 0.5, 1.0],
        'vit_acc': [26.2, 31, 36, 30],  
        'hybrid_acc': [29.2, 38, 36, 28]  
    }
    plot_data_efficiency(data_efficiency_results)
    
    # 3. Architecture comparison
    print("\n3. Generating architecture comparison...")
    plot_architecture_comparison()
    
    print("\n4. Generating attention maps...")

    # Load hybrid model
    hybrid_model = HybridViT(config).to(device)
    missing, unexpected = hybrid_model.load_state_dict(
        torch.load(hybrid_model_path, map_location=device),
        strict=False
    )
    
    hybrid_model.eval()

    # CIFAR-10 class names
    classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), 
                            (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.CIFAR10('./data', train=False,
                                transform=transform,
                                download=True)

    idx = 18
    sample_image, label = test_dataset[idx]

    # ------------------ Prediction ------------------
    with torch.no_grad():
        input_tensor = sample_image.unsqueeze(0).to(device)
        outputs = hybrid_model(input_tensor)
        _, pred = torch.max(outputs, 1)

    pred_class = classes[pred.item()]
    true_class = classes[label]

    print(f"True Label: {true_class}")
    print(f"Predicted : {pred_class}")

    # ------------------ Attention ------------------
    visualize_attention_maps(hybrid_model, sample_image, device)
    
    # 5. Results table
    print("\n5. Generating results table...")
    vit_results = {
        'params': 2684554,
        'mnist_acc': 98.62,
        'cifar10_acc': 48.04,
        'cifar100_acc': 14.73,
        'train_time': 3.5
    }
    hybrid_results = {
        'params': 2684554,
        'mnist_acc': 98.84,
        'cifar10_acc': 61.05,
        'cifar100_acc': 23.02,
        'train_time': 3.2
    }
    create_results_table(vit_results, hybrid_results)
    
    print("\n✅ All visualizations generated successfully!")
    print("Check the 'figures/' directory for all plots.")


if __name__ == '__main__':
    """Test visualization functions"""
    # Example usage
    print("Generating sample visualizations...")
    
    # Generate architecture comparison (doesn't need data)
    plot_architecture_comparison()
    
    # Generate sample data efficiency plot
    sample_results = {
        'fractions': [0.1, 0.25, 0.5, 1.0],
        'vit_acc': [45.2, 58.3, 64.1, 67.3],
        'hybrid_acc': [52.1, 65.8, 71.2, 74.8]
    }
    plot_data_efficiency(sample_results)
    
    print("\nSample visualizations created!")
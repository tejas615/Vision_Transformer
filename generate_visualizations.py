"""
Main script to generate all visualizations for the presentation.

Usage:
    python scripts/generate_visualizations.py
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.visualize import generate_all_visualizations
from configs.cifar_config import CIFAR10Config


def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("Generating Visualizations for ViT vs Hybrid Presentation")
    print("=" * 60)
    
    # Configuration
    config = CIFAR10Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths to trained models
    vit_model_path = 'experiments/vit_cifar10_best.pth'
    hybrid_model_path = 'experiments/hybrid_cifar10_best.pth'
    
    # Check if models exist
    if not Path(vit_model_path).exists():
        print(f"\n❌ ViT model not found at {vit_model_path}")
        print("Please train the model first using: python scripts/train_cifar10.py --model vit")
        return
    
    if not Path(hybrid_model_path).exists():
        print(f"\n❌ Hybrid model not found at {hybrid_model_path}")
        print("Please train the model first using: python scripts/train_cifar10.py --model hybrid")
        return
    
    # Generate all visualizations
    generate_all_visualizations(
        vit_model_path=vit_model_path,
        hybrid_model_path=hybrid_model_path,
        config=config,
        device=device
    )
    
    print("\n" + "=" * 60)
    print("✅ All visualizations generated successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  • figures/training_curves.png")
    print("  • figures/data_efficiency.png")
    print("  • figures/architecture_comparison.png")
    print("  • figures/attention_maps.png")
    print("  • figures/results_table.png")
    print("\nUse these in your presentation!")


if __name__ == '__main__':
    main()
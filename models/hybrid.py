"""
Hybrid CNN-ViT Implementation

This module implements a hybrid Vision Transformer that uses convolutional
layers for patch embedding instead of linear projection. This provides
inductive biases (locality, translation equivariance) that improve
data efficiency on small datasets.

Key differences from standard ViT:
- CNN-based patch embedding with hierarchical feature extraction
- Maintains same transformer encoder architecture
- Better performance on small datasets (CIFAR-10, CIFAR-100)
"""

import torch.nn as nn
import torch
from models.vit import TransformerBlock


class CNNPatchEmbed(nn.Module):
    
    """
    Convolutional Patch Embedding.
    
    Replaces the simple linear projection in ViT with a convolutional
    feature extractor. This provides inductive biases:
    - Locality: Small conv filters process local regions
    - Translation equivariance: Same filters applied everywhere
    - Hierarchical features: Progressive feature extraction
    
    Architecture:
        Input (32×32×3)
          → Conv1 (3→64, 3×3, stride=2) → 16×16×64
          → BatchNorm + GELU
          → Conv2 (64→128, 3×3, stride=2) → 8×8×128
          → BatchNorm + GELU
          → Conv3 (128→embed_dim, 1×1) → 8×8×embed_dim
          → Flatten → num_patches × embed_dim
    
    Args:
        img_size (int): Input image size. Default: 32
        patch_size (int): Target patch size (determines final feature map size). Default: 4
        in_channels (int): Number of input channels. Default: 3
        embed_dim (int): Output embedding dimension. Default: 384
        
    Attributes:
        num_patches (int): Number of output patches
        conv_stem (nn.Sequential): Convolutional feature extractor
        
    Example:
        >>> cnn_embed = CNNPatchEmbed(img_size=32, patch_size=4, embed_dim=384)
        >>> x = torch.randn(2, 3, 32, 32)
        >>> patches = cnn_embed(x)
        >>> print(patches.shape)  # torch.Size([2, 64, 384])
    """
    
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolutional stem for feature extraction
        # We use two stride=2 convolutions to achieve patch_size=4 reduction
        # 32 → 16 → 8 (if patch_size=4)
        self.conv_stem = nn.Sequential(
            # First convolutional block
            # Captures basic features: edges, colors
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            # Second convolutional block
            # Captures more complex patterns: textures, simple shapes
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            # Final projection to embedding dimension
            # 1×1 convolution adjusts channels without changing spatial size
            nn.Conv2d(128, embed_dim, kernel_size=1)
        )
        
        # This achieves patch_size=4 reduction (32→8 via two stride-2 convs)
    
    def forward(self, x):
        """
        Forward pass of CNN patch embedding.
        
        Args:
            x (torch.Tensor): Input images of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Patch embeddings of shape [B, num_patches, embed_dim]
        """
        
        # Apply convolutional stem
        # [B, C, H, W] -> [B, embed_dim, H/patch_size, W/patch_size]
        x = self.conv_stem(x)  # [B, embed_dim, H/4, W/4]
        
        # Flatten spatial dimensions
        # [B, embed_dim, H/P, W/P] -> [B, embed_dim, num_patches]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        
        # Transpose to get patch sequence
        # [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x
    


class HybridViT(nn.Module):
    """
    Hybrid CNN-Vision Transformer.
    
    Combines CNN inductive biases with Transformer's global modeling capability.
    Uses convolutional layers for patch embedding while maintaining the same
    transformer encoder architecture as standard ViT.
    
    Why Hybrid Outperforms Pure ViT on Small Datasets:
    
    1. **Locality Bias**: CNNs naturally process local neighborhoods with small
       filters (3×3), capturing edges and textures efficiently. ViT must learn
       this from scratch.
       
    2. **Translation Equivariance**: CNN weight sharing means a "cat detector"
       works everywhere in the image. ViT must learn separate representations
       for cats in different positions.
       
    3. **Hierarchical Features**: CNN layers progressively build features:
       edges → textures → patterns. ViT receives flat patches and must learn
       this hierarchy from data.
    
    4. **Data Efficiency**: With these built-in biases, the hybrid needs fewer
       examples to generalize. ViT requires massive datasets to learn what
       CNNs know by design.
    
    Args:
        config: Configuration object with attributes:
            - img_size, patch_size, in_channels, num_classes
            - embed_dim, num_layers, num_heads, mlp_ratio, dropout
            
    Example:
        >>> from configs.config import CIFAR10Config
        >>> config = CIFAR10Config()
        >>> model = HybridViT(config)
        >>> x = torch.randn(4, 3, 32, 32)
        >>> logits = model(x)
        >>> print(logits.shape)  # torch.Size([4, 10])
    """
    
    def __init__(self, config):
        super().__init__()
        # CNN-based patch embedding (our innovation!)
        self.patch_embed = CNNPatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=1 if config.dataset == 'MNIST' else 3,
            embed_dim=config.embed_dim
        )
        
        # Rest is same as ViT
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, 
                           config.mlp_ratio, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        
        # Initialize model weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        """
        Forward pass of Hybrid ViT.
        
        The only difference from ViT is the patch embedding step:
        - ViT: Linear projection of flattened patches
        - Hybrid: CNN feature extraction before flattening
        
        Args:
            x (torch.Tensor): Input images [B, C, H, W]
            
        Returns:
            torch.Tensor: Class logits [B, num_classes]
        """
        
        B = x.shape[0]
        
        # CNN patch embedding (different from ViT!)
        x = self.patch_embed(x) # [B, num_patches, embed_dim]
        
        # Everything else is identical to ViT
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x
"""
Vision Transformer (ViT) Implementation

This module implements the Vision Transformer architecture from the paper:
"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
by Dosovitskiy et al., ICLR 2021.

The implementation includes:
- Patch embedding layer
- Transformer encoder blocks
- Classification head
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding using a convolutional layer.
    
    Splits the input image into fixed-size patches and projects them
    to an embedding dimension using a convolution operation.
    
    Args:
        img_size (int): Size of input image (assumes square images). Default: 224
        patch_size (int): Size of each patch (assumes square patches). Default: 16
        in_channels (int): Number of input channels. Default: 3 (RGB)
        embed_dim (int): Embedding dimension. Default: 768
        
    Attributes:
        num_patches (int): Total number of patches in the image
        proj (nn.Conv2d): Convolutional projection layer
        
    Example:
        >>> patch_embed = PatchEmbed(img_size=224, patch_size=16, embed_dim=768)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> patches = patch_embed(x)
        >>> print(patches.shape)  # torch.Size([2, 196, 768])
    """
    
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # Use convolution for patch embedding
        # kernel_size=patch_size, stride=patch_size creates non-overlapping patches
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        """
        Forward pass of patch embedding.
        
        Args:
            x (torch.Tensor): Input images of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Patch embeddings of shape [B, num_patches, embed_dim]
        """
        
        # Apply convolution: [B, C, H, W] -> [B, embed_dim, H/P, W/P]
        x = self.proj(x)  
        # Flatten spatial dimensions: [B, embed_dim, H/P, W/P] -> [B, embed_dim, num_patches]
        x = x.flatten(2)  
        # Transpose to get: [B, num_patches, embed_dim]
        x = x.transpose(1, 2)  
        return x
    
class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block.
    
    Implements a single transformer encoder layer with multi-head self-attention
    and feed-forward MLP, following the architecture from "Attention is All You Need"
    with pre-normalization from ViT.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim. Default: 4.0
        dropout (float): Dropout probability. Default: 0.1
        
    Attributes:
        norm1 (nn.LayerNorm): Layer normalization before attention
        attn (nn.MultiheadAttention): Multi-head self-attention layer
        norm2 (nn.LayerNorm): Layer normalization before MLP
        mlp (nn.Sequential): Feed-forward network
        Example:
        >>> block = TransformerBlock(embed_dim=768, num_heads=12)
        >>> x = torch.randn(2, 197, 768)  # [batch, tokens, embed_dim]
        >>> output = block(x)
        >>> print(output.shape)  # torch.Size([2, 197, 768])
    """
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        # Pre-normalization (LayerNorm before attention/MLP)
        self.norm1 = nn.LayerNorm(embed_dim)
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True) # Input/output shape: [B, N, E]
        self.norm2 = nn.LayerNorm(embed_dim)
        # Feed-forward network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(), # Gaussian Error Linear Unit activation
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Forward pass of transformer block.
        
        Uses pre-normalization and residual connections:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, embed_dim]
            
        Returns:
            torch.Tensor: Output tensor of shape [B, N, embed_dim]
        """
        
        # Multi-head self-attention with residual connection
        # attn returns (output, attention_weights), we only need output
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # Feed-forward network with residual connection
        x = x + self.mlp(self.norm2(x))
        return x
    
class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model.
    
    Implements the full Vision Transformer architecture for image classification.
    The model processes images by:
    1. Splitting into patches and linearly embedding them
    2. Adding learnable position embeddings
    3. Prepending a learnable [CLS] token
    4. Processing through transformer encoder layers
    5. Classifying based on the [CLS] token output
    
    Args:
        config: Configuration object with the following attributes:
            - img_size (int): Input image size
            - patch_size (int): Patch size
            - in_channels (int): Number of input channels (1 for MNIST, 3 for CIFAR)
            - num_classes (int): Number of output classes
            - embed_dim (int): Embedding dimension
            - num_layers (int): Number of transformer blocks
            - num_heads (int): Number of attention heads
            - mlp_ratio (float): MLP hidden dim ratio
            - dropout (float): Dropout probability
            
        Attributes:
        patch_embed (PatchEmbed): Patch embedding layer
        cls_token (nn.Parameter): Learnable classification token
        pos_embed (nn.Parameter): Learnable position embeddings
        blocks (nn.ModuleList): List of transformer encoder blocks
        norm (nn.LayerNorm): Final layer normalization
        head (nn.Linear): Classification head
        
    Example:
        >>> from configs.config import CIFAR10Config
        >>> config = CIFAR10Config()
        >>> model = VisionTransformer(config)
        >>> x = torch.randn(4, 3, 32, 32)
        >>> logits = model(x)
        >>> print(logits.shape)  # torch.Size([4, 10])
    """
    
    def __init__(self, config):
        super().__init__()
        # Patch embedding layer
        self.config = config
        self.patch_embed = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=1 if config.dataset == 'MNIST' else 3,
            embed_dim=config.embed_dim
        )
        
        # Learnable [CLS] token
        # Shape: [1, 1, embed_dim], will be expanded to [B, 1, embed_dim]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        
        # Position embeddings
        num_patches = self.patch_embed.num_patches
        # Learnable position embeddings
        # Shape: [1, num_patches + 1, embed_dim]
        # +1 for the [CLS] token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, 
                           config.mlp_ratio, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(config.embed_dim)
        # Classification head
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        
        """
        Initialize model weights.
        
        Uses truncated normal distribution for positional embeddings and CLS token,
        following the original ViT implementation.
        """
        # Truncated normal initialization for position embeddings and CLS token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Xavier uniform for other linear layers (done automatically by PyTorch)
    
    def forward(self, x):
        """
        Forward pass of Vision Transformer.
        
        Processing steps:
        1. Patch embedding: [B, C, H, W] -> [B, num_patches, embed_dim]
        2. Prepend [CLS] token: [B, num_patches+1, embed_dim]
        3. Add position embeddings
        4. Pass through transformer blocks
        5. Extract [CLS] token and classify
        
        Args:
            x (torch.Tensor): Input images of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Class logits of shape [B, num_classes]
        """
        B = x.shape[0]
        
        # Patch embedding: [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Expand and prepend [CLS] token
        # cls_token: [1, 1, embed_dim] -> [B, 1, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        
        # Concatenate [CLS] token with patch embeddings
        # [B, num_patches, embed_dim] -> [B, num_patches+1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]
        
        # Add position embeddings
        # Both x and pos_embed have shape [B, num_patches+1, embed_dim]
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final layer normalization
        x = self.norm(x[:, 0])  # [B, embed_dim]
        x = self.head(x)  # [B, num_classes]
        
        return x
    
from configs.config import Config
config = Config()
model = VisionTransformer(config)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Test forward pass
x = torch.randn(4, 1, 28, 28)
out = model(x)
print(out.shape)  # Should be [4, 10]

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # Simple linear projection
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Pre-norm architecture (from ViT paper)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=1 if config.dataset == 'MNIST' else 3,
            embed_dim=config.embed_dim
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        
        # Position embeddings
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, 
                           config.mlp_ratio, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        
        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Get CLS token output
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

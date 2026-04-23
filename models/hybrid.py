import torch.nn as nn
import torch
from models.vit import TransformerBlock


class CNNPatchEmbed(nn.Module):
    """Replace simple linear projection with CNN feature extractor"""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolutional stem (instead of single projection)
        self.conv_stem = nn.Sequential(
            # First conv: 3→64, stride=2, reduces 32→16
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            # Second conv: 64→128, stride=2, reduces 16→8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            # Final projection to embed_dim
            nn.Conv2d(128, embed_dim, kernel_size=1)
        )
        
        # This achieves patch_size=4 reduction (32→8 via two stride-2 convs)
    
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.conv_stem(x)  # [B, embed_dim, H/4, W/4]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x
    


class HybridViT(nn.Module):
    """ViT with CNN patch embedding"""
    def __init__(self, config):
        super().__init__()
        # Use CNN patch embedding instead of linear
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
        
        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x
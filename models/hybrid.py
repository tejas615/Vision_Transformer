import torch.nn as nn

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
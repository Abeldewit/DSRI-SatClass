import torch
import torch.nn as nn


class TemporalPatchEmb(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        embed_dim,
        channels,
        time_steps=61,
        ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.channels = channels

        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")

        self.grid_size = self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv3d(
            time_steps, 
            embed_dim, 
            kernel_size=(channels, patch_size, patch_size), 
            stride=(channels, patch_size, patch_size)
        )
        
        self.proj.requires_grad_(False)
    
    def forward(self, x):
        # We get a tensor of shape (B, T, C, H, W)
        # Where T is the number of frames
        x = self.proj(x)
        return x

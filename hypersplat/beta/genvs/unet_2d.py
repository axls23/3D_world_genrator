
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class DiffusionUNet(nn.Module):
    """
    2D U-Net (ADM / DDPM++).
    Conditioned on 3D Feature Image via Concatenation.
    
    Paper Spec:
    - Input: 19 Channels (3 RGB + 16 Feature)
    - Output: 3 Channels (Noise)
    - Base Channels: 128
    - Ch Multipliers: 1, 2, 2, 2
    - Attention: 16, 8 resolutions
    - Params: ~90M
    """
    
    def __init__(self, 
                 in_channels=19, 
                 out_channels=3, 
                 model_channels=128, 
                 channel_mult=(1, 2, 2, 2),
                 attention_resolutions=(16, 8)):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        
        # Time Embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Encoder (Downsampling)
        self.input_blocks = nn.ModuleList()
        # Initial Conv
        self.input_blocks.append(nn.Conv2d(in_channels, model_channels, 3, padding=1))
        
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(2): # 2 ResBlocks per level
                layers = [ResBlock(ch, time_embed_dim, dropout=0, out_channels=mult * model_channels)]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
                
            if level != len(channel_mult) - 1:
                self.input_blocks.append(Downsample(ch))
                input_block_chans.append(ch)
                ds *= 2
        
        # Middle
        self.middle_block = nn.Sequential(
            ResBlock(ch, time_embed_dim, dropout=0),
            AttentionBlock(ch),
            ResBlock(ch, time_embed_dim, dropout=0),
        )
        
        # Decoder (Upsampling)
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(2 + 1): # 3 blocks per level (usually)
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim, dropout=0, out_channels=mult * model_channels)]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                    
                if level and i == 2:
                    layers.append(Upsample(ch))
                    ds //= 2
                    
                self.output_blocks.append(nn.Sequential(*layers))
                
        # Final
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, feature_cond):
        """
        x: [B, 3, H, W] (Noisy Image)
        feature_cond: [B, 16, H, W] (Rendered Feature Image)
        """
        # Concatenate conditioning (Section 5.2)
        h = torch.cat([x, feature_cond], dim=1)
        
        # Time embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        hs = []
        # Down
        for module in self.input_blocks:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, emb)
                    else:
                        h = layer(h)
            else:
                h = module(h)
            hs.append(h)
        
        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
            else:
                h = layer(h)
        
        # Up
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, emb)
                    else:
                        h = layer(h)
            else:
                h = module(h)
            
        return self.out(h)

# Helpers
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class ResBlock(nn.Module):
    def __init__(self, channels, emb_channels, dropout, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.out_channels),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
        )
        if channels != self.out_channels:
            self.skip = nn.Conv2d(channels, self.out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        h = h + emb_out[..., None, None]
        h = self.out_layers(h)
        return self.skip(x) + h

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.reshape(B, C, -1).permute(0, 2, 1)
        k = k.reshape(B, C, -1) # B, C, N
        v = v.reshape(B, C, -1).permute(0, 2, 1)
        
        attn = torch.bmm(q, k) * (int(C) ** (-0.5))
        attn = F.softmax(attn, dim=-1)
        
        h = torch.bmm(attn, v)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj_out(h)

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    def forward(self, x): return self.op(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

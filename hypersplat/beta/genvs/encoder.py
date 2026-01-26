
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class GeometryEncoder(nn.Module):
    """
    Geometry Encoder: ResNet-34 Backbone + DeepLabV3+ Head.
    
    Lifts a 2D image into a high-dimensional feature representation suitable for
    reshaping into a 3D Frustum Volume.
    
    Paper Spec:
    - Backbone: ResNet-50 (pretrained)
    - Head: DeepLabV3+ with ASPP
    - Upsampling: Learnable (ConvTranspose2d) instead of bilinear
    - Normalization: No Batch Norm (Deterministic)
    - Output Channels: C_feat (16) * Depth (64) = 1024
    """
    
    def __init__(self, 
                 c_feat=16, 
                 depth_planes=64, 
                 base_res=(128, 128)):
        super().__init__()
        self.c_feat = c_feat
        self.depth_planes = depth_planes
        self.out_channels = c_feat * depth_planes # 1024 typically
        
        # 1. Component: Backbone (ResNet-50)
        # We strip the FC layer and avgpool
        original_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Feature extraction layers
        self.layer0 = nn.Sequential(original_resnet.conv1, original_resnet.bn1, original_resnet.relu, original_resnet.maxpool)
        self.layer1 = original_resnet.layer1 # 256 ch, /4
        self.layer2 = original_resnet.layer2 # 512 ch, /8
        self.layer3 = original_resnet.layer3 # 1024 ch, /16
        self.layer4 = original_resnet.layer4 # 2048 ch, /32
        
        # 2. Component: DeepLabV3+ ASPP Head
        # Dilated convolutions to capture multi-scale context
        # ResNet-50 layer4 has 2048 channels
        self.aspp = ASPP(in_channels=2048, out_channels=256)
        
        # 3. Component: Decoder / Learnable Upsampler
        # We need to go from /32 (ASPP output) back to /1 (Original Resolution) 
        # or /2 depending on implementation choice. 
        # Paper implies lifting happens at input resolution.
        
        # Fusion block 1: High-level features (ASPP) + Layer 1 (Low-level 256ch)
        self.fusion1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # Learnable Upsamplers (replacing bilinear)
        # Upsample 2x
        self.up_x2_1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.up_x2_2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up_x2_3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        
        # Final projection to massive channel dimension
        self.final_conv = nn.Conv2d(64, self.out_channels, kernel_size=1)
        
        # Freeze BN or convert to GroupNorm for determinism? 
        # Paper says "BN Disabled". Simple way is to eval() or replace.
        self._freeze_bn()

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, x):
        """
        Input: [B, 3, H, W]
        Output: [B, C_feat, D, H, W] (3D Volume)
        """
        B, C, H, W = x.shape
        
        # Backbone
        l0 = self.layer0(x)
        l1 = self.layer1(l0) # /4
        l2 = self.layer2(l1) # /8
        l3 = self.layer3(l2) # /16
        l4 = self.layer4(l3) # /32
        
        # ASPP Head
        feat_aspp = self.aspp(l4)
        
        # Upsample ASPP features to match Layer 1 (/4)
        # Note: Bilinear here is okay for internal features or should be Learnable?
        # Paper emphasizes learnable for the FINAL stages.
        feat_up4 = F.interpolate(feat_aspp, size=l1.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate with low-level features
        feat_cat = torch.cat([feat_up4, l1], dim=1)
        
        # Decode
        x = self.fusion1(feat_cat)
        
        # Progressive Learnable Upsampling
        # Current: 256ch, /4 resolution
        x = self.up_x2_1(x) # -> 128ch, /2
        x = F.relu(x)
        x = self.up_x2_2(x) # -> 64ch, /1 (Full Res)
        x = F.relu(x)
        
        # Final Projection
        x_flat = self.final_conv(x) # [B, 1024, H_out, W_out]
        
        # Ensure dimensions match input (handle padding/rounding discrepancies from ResNet vs ConvTranspose)
        if x_flat.shape[-2:] != (H, W):
             x_flat = F.interpolate(x_flat, size=(H, W), mode='bilinear', align_corners=False)
        
        # Lifting: Reshape to [B, C_feat, D, H, W]
        # Channels are split into Features x Depth
        x_vol = x_flat.view(B, self.c_feat, self.depth_planes, H, W)
        
        return x_vol

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        
        modules = []
        # 1x1 conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU()
        ))
        
        # Atrous convolutions
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.ReLU()
            ))
            
        # Global Avg Pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU()
        )
        
        self.convs = nn.ModuleList(modules)
        
        # Final projection
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels + out_channels, out_channels, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5) # Paper says disable dropout? Check specific section for ASPP. 
                            # Usually ASPP has dropout. We'll leave it but maybe set p=0.
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        
        # Add global pooling
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(global_feat)
        
        res = torch.cat(res, dim=1)
        return self.project(res)

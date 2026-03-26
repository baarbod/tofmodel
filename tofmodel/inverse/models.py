import torch
import torch.nn as nn

class ResBlock1D(nn.Module):
    def __init__(self, dim, kernel_size=5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(dim)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x)) # Residual connection

class TOFinverse(nn.Module):
    def __init__(self, nflow_in, nfeature_out, context_dim=32, hidden_dim=128, num_blocks=4):
        """
        Deep dual-input network for TOF inversion.
        num_blocks: Increase this to make the network deeper (e.g., 4, 6, 8).
        """
        super().__init__()
        
        # 1. Flow Feature Extractor
        self.flow_in = nn.Sequential(
            nn.Conv1d(nflow_in, hidden_dim, kernel_size=5, padding='same'),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # 2. Area Context Extractor
        self.area_in = nn.Sequential(
            nn.Conv1d(1, context_dim, kernel_size=5, padding='same'),
            nn.BatchNorm1d(context_dim),
            nn.GELU()
        )
        
        # 3. Deep Processing (Combined Features)
        combined_dim = hidden_dim + context_dim
        
        # Stack residual blocks for depth without vanishing gradients
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResBlock1D(combined_dim))
        self.deep_processor = nn.Sequential(*blocks)
        
        # 4. Final Projection to Velocity
        self.out_conv = nn.Sequential(
            nn.Conv1d(combined_dim, hidden_dim // 2, kernel_size=3, padding='same'),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, nfeature_out, kernel_size=3, padding='same')
        )

    def forward(self, flow_x, area_x):
        # Extract initial features
        f_feat = self.flow_in(flow_x)
        a_feat = self.area_in(area_x)
        
        # Concatenate along the channel dimension
        combined = torch.cat([f_feat, a_feat], dim=1)
        
        # Pass through deep residual blocks
        deep_feat = self.deep_processor(combined)
        
        # Predict output
        return self.out_conv(deep_feat)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class FiLMResDilatedBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, dilation, context_dim=32):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation)
        
#         # FiLM projection: maps global context to scale (gamma) and shift (beta)
#         self.film_proj = nn.Linear(context_dim, 2 * out_ch)
        
#         self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

#     def forward(self, x, context):
#         res = self.shortcut(x)
        
#         out = self.conv1(x)
        
#         # Spatial FiLM Modulation
#         film_params = self.film_proj(context).unsqueeze(-1) 
#         gamma, beta = torch.chunk(film_params, 2, dim=1) 
        
#         # (1 + gamma) ensures identity scaling if weights are initialized near 0
#         out = out * (1 + gamma) + beta 
#         out = self.relu(out)
        
#         out = self.conv2(out)
#         return self.relu(out + res)

# class GeometryEncoder(nn.Module):
#     def __init__(self, in_ch=1, context_dim=32):
#         super().__init__()
#         # Condenses the spatial profile into a 1D context vector
#         self.net = nn.Sequential(
#             nn.Conv1d(in_ch, 8, kernel_size=5, stride=2, padding=2),
#             nn.ReLU(),
#             nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1) 
#         )
#         self.proj = nn.Linear(16, context_dim)

#     def forward(self, x):
#         # x shape: [Batch, 1, 300]
#         features = self.net(x).squeeze(-1) 
#         return self.proj(features) 

# class TOFinverse(nn.Module):
#     def __init__(self, nflow_in=3, nfeature_out=1, context_dim=32):
#         super().__init__()
        
#         # 1. SPATIAL GEOMETRY ENCODER
#         self.geom_encoder = GeometryEncoder(in_ch=1, context_dim=context_dim)
        
#         # 2. DILATED FEATURE EXTRACTOR
#         self.layer1 = FiLMResDilatedBlock(nflow_in, 16, dilation=1, context_dim=context_dim)
#         self.layer2 = FiLMResDilatedBlock(16, 32, dilation=2, context_dim=context_dim)
#         self.layer3 = FiLMResDilatedBlock(32, 64, dilation=4, context_dim=context_dim)
#         self.layer4 = FiLMResDilatedBlock(64, 32, dilation=8, context_dim=context_dim)
#         self.layer5 = FiLMResDilatedBlock(32, 16, dilation=16, context_dim=context_dim)
        
#         self.out_head = nn.Conv1d(16, nfeature_out, kernel_size=1)
        
#         # 3. PHYSICS SKIP CONNECTION
#         self.physics_skip = nn.Conv1d(nflow_in, nfeature_out, kernel_size=1)
        
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
#             nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, flow_x, area_x):
#         context = self.geom_encoder(area_x)
        
#         h = self.layer1(flow_x, context)
#         h = self.layer2(h, context)
#         h = self.layer3(h, context)
#         h = self.layer4(h, context)
#         h = self.layer5(h, context)
#         deep_out = self.out_head(h)
        
#         phys_out = self.physics_skip(flow_x)
        
#         return deep_out + phys_out
    
    

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ResDilatedBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, dilation):
#         super().__init__()
#         # Use 'same' padding to keep the temporal dimensions aligned
#         self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, 
#                                padding=dilation, dilation=dilation)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, 
#                                padding=dilation, dilation=dilation)
        
#         # Identity shortcut to match channel dimensions
#         self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

#     def forward(self, x):
#         res = self.shortcut(x)
#         x = self.relu(self.conv1(x))
#         x = self.conv2(x)
#         return self.relu(x + res)

# class TOFinverse(nn.Module):
#     def __init__(self, nfeature_in=5, nfeature_out=1, input_size=None, output_size=None):
#         super().__init__()
        
#         # 1. LEARNABLE INPUT SCALING
#         # This bridges the gap between your 0-1 fMRI ratios and physical unit area curves
#         # without destroying the raw data distribution.
#         self.input_scaler = nn.Parameter(torch.ones(1, nfeature_in, 1))
#         self.input_bias = nn.Parameter(torch.zeros(1, nfeature_in, 1))

#         # 2. DILATED FEATURE EXTRACTOR
#         # Receptive Field: ~125 points (approx 47s), enough for 0.025Hz cycles
#         self.layer1 = ResDilatedBlock(nfeature_in, 16, dilation=1)
#         self.layer2 = ResDilatedBlock(16, 32, dilation=2)
#         self.layer3 = ResDilatedBlock(32, 64, dilation=4)
#         self.layer4 = ResDilatedBlock(64, 32, dilation=8)
#         self.layer5 = ResDilatedBlock(32, 16, dilation=16)
        
#         # Final non-linear projection
#         self.out_head = nn.Conv1d(16, nfeature_out, kernel_size=1)
        
#         # 3. PHYSICS SKIP CONNECTION
#         # Directly maps the 3 fMRI slices to velocity. This ensures that the 
#         # relative amplitude information is always a primary driver of the output.
#         self.physics_skip = nn.Conv1d(3, nfeature_out, kernel_size=1)
        
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Conv1d):
#             nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         # x shape: [Batch, 5, Time]
#         # Ch 0,1,2: fMRI | Ch 3,4: Area/Depth (Physical Units)
        
#         # Apply learnable scaling
#         x_scaled = x * self.input_scaler + self.input_bias
        
#         # Deep Path
#         h = self.layer1(x_scaled)
#         h = self.layer2(h)
#         h = self.layer3(h)
#         h = self.layer4(h)
#         h = self.layer5(h)
#         deep_out = self.out_head(h)
        
#         # Physics Path (using original fMRI slices only)
#         phys_out = self.physics_skip(x[:, :3, :])
        
#         # Combine: Deep path learns the modulation/lags, Phys path anchors the amplitude
#         return deep_out + phys_out

    
    
    
    
    
# import torch.nn as nn


# class Conv(nn.Module):
#     def __init__(self, in_channels, out_channels, dilation=1):
#         super().__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels,
#                               kernel_size=3, padding=dilation, dilation=dilation)
#         self.norm = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         return self.relu(self.norm(self.conv(x)))


# class TOFinverse(nn.Module):
#     def __init__(self, nfeature_in, nfeature_out, input_size, output_size):
#         super().__init__()

#         self.down1 = Conv(nfeature_in, 16, dilation=1)
#         self.down2 = Conv(16, 32, dilation=2)
#         self.down3 = Conv(32, 64, dilation=4)
#         self.up1 = Conv(64, 32, dilation=2)
#         self.up2 = Conv(32, 16, dilation=1)
#         self.out = nn.Conv1d(16, nfeature_out, kernel_size=1)

#         self.output_size = output_size
#         self.nfeature_out = nfeature_out

#     def forward(self, x):
#         x = self.down1(x)
#         x = self.down2(x)
#         x = self.down3(x)
#         x = self.up1(x)
#         x = self.up2(x)
#         x = self.out(x)
#         return x

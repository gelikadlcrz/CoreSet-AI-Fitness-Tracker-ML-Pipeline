import torch
import torch.nn as nn
from src.utils.graph_utils import build_blazepose_33_adjacency_matrix
from src.models.gcn import ST_GCN_Block
import torch.nn.functional as F

class CoreSetSTGCN_MultiTask(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, max_frames=150, node_count=33):
        super().__init__()
        
        self.num_classes = num_classes
        self.max_frames = max_frames
        
        adj = build_blazepose_33_adjacency_matrix()
        
        # Learned 1D temporal smoothing at input stage
        self.input_smoothing = nn.Conv2d(in_channels, in_channels, kernel_size=(9, 1), padding=(4, 0))
        
        # Backbone
        self.st_gcn_1 = ST_GCN_Block(in_channels, out_channels=32, adj_matrix=adj, residual=False)
        self.batch_norm_1 = nn.BatchNorm3d(32) 
        self.st_gcn_2 = ST_GCN_Block(32, 64, adj, stride=2)
        self.st_gcn_3 = ST_GCN_Block(64, 128, adj, stride=2)
        self.global_pooling = nn.AdaptiveAvgPool3d(1) 
        
        # Head 1: Classification
        self.head_classification = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # Head 2: Density Map
        self.head_density_map = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv1d(128, 1, kernel_size=1) 
        )

    def forward(self, x):
        b, c, t, v, m = x.size()
        
        # Apply 1D Temporal Smoothing
        x_safe = x.permute(0, 4, 1, 2, 3).contiguous()
        x_reshaped = x_safe.view(b * m, c, t, v)
        x_smoothed = self.input_smoothing(x_reshaped)
        x = x_smoothed.view(b, m, c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        
        # Pass to ST-GCN blocks
        x = self.batch_norm_1(self.st_gcn_1(x))
        x = self.st_gcn_2(x)
        x = self.st_gcn_3(x) 
        
        # --- Classification Path ---
        classification_map = self.global_pooling(x) 
        classification_features = classification_map.view(b, classification_map.size(1))
        classification_logits = self.head_classification(classification_features)
        
        # --- Density Map Path ---
        density_spatial_pool = nn.AdaptiveAvgPool3d((x.size(2), 1, 1)) 
        density_temporal_series = density_spatial_pool(x) 
        
        density_series_1d = density_temporal_series.view(b, density_temporal_series.size(1), density_temporal_series.size(2)) 
        
        # Predict the compressed density map
        density_map_raw = self.head_density_map(density_series_1d) 
        
        # NEW: Interpolate it back to the exact original frame count (150) for accurate MSE Loss calculation
        density_map_upsampled = F.interpolate(density_map_raw, size=self.max_frames, mode='linear', align_corners=False)
        density_map_output = density_map_upsampled.view(b, self.max_frames) 
        density_map_prob = torch.sigmoid(density_map_output)
        
        return classification_logits, density_map_prob
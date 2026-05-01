import torch
import torch.nn as nn
from src.utils.graph_utils import build_blazepose_33_adjacency_matrix
from src.models.gcn import ST_GCN_Block

class CoreSetSTGCN_MultiTask(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, max_frames=150, node_count=33):
        super().__init__()
        
        self.num_classes = num_classes
        self.max_frames = max_frames
        
        adj = build_blazepose_33_adjacency_matrix()
        
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
        
        x = self.batch_norm_1(self.st_gcn_1(x))
        x = self.st_gcn_2(x)
        x = self.st_gcn_3(x) 
        
        # Classification Path
        classification_map = self.global_pooling(x) 
        classification_features = classification_map.view(b, classification_map.size(1))
        classification_logits = self.head_classification(classification_features)
        
        # Density Map Path
        density_features = x.permute(0, 1, 2, 3, 4).contiguous() 
        density_spatial_pool = nn.AdaptiveAvgPool3d((t//4, 1, 1)) 
        density_temporal_series = density_spatial_pool(x) 
        
        density_series_1d = density_temporal_series.view(b, density_temporal_series.size(1), density_temporal_series.size(2)) 
        
        density_map_raw = self.head_density_map(density_series_1d) 
        density_map_output = density_map_raw.view(b, density_map_raw.size(2)) 
        density_map_prob = torch.sigmoid(density_map_output)
        
        return classification_logits, density_map_prob
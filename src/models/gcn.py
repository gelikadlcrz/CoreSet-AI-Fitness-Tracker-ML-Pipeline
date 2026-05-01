import torch
import torch.nn as nn
import numpy as np

class SpatialGraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, norm_adj_matrix, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.register_buffer('adj', torch.from_numpy(norm_adj_matrix).float())
        # FIX: Using PyTorch's optimized 1x1 Conv2D instead of fragile manual weight parameters
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, t, v, m = x.size()
        
        # Safely fold the Persons (m) dimension into the Batch (b) dimension
        x_safe = x.permute(0, 4, 1, 2, 3).contiguous() # (b, m, c, t, v)
        x_reshaped = x_safe.view(b * m, c, t, v)
        
        # 1. Feature Transformation (X * W)
        x_transformed = self.conv(x_reshaped) # (b*m, out_c, t, v)
        
        # 2. Graph Aggregation (A * X)
        # matmul automatically broadcasts over the batch and time dimensions safely
        output = torch.matmul(x_transformed, self.adj)
        
        # Safely unfold the Persons dimension back out
        output = output.view(b, m, self.out_channels, t, v)
        output = output.permute(0, 2, 3, 4, 1).contiguous() # (b, out_c, t, v, m)
        return output

class TemporalConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2 
        self.t_conv = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=(kernel_size, 1), 
                                stride=(stride, 1), 
                                padding=(padding, 0))

    def forward(self, x):
        b, c, t, v, m = x.size()
        
        # Safe memory folding
        x_safe = x.permute(0, 4, 1, 2, 3).contiguous()
        x_reshaped = x_safe.view(b * m, c, t, v) 
        
        output = self.t_conv(x_reshaped) 
        
        # Safe memory unfolding
        output = output.view(b, m, output.size(1), output.size(2), output.size(3))
        output = output.permute(0, 2, 3, 4, 1).contiguous() 
        return output

class ST_GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, adj_matrix, t_kernel_size=9, stride=1, residual=True):
        super().__init__()
        self.gcn = SpatialGraphConvolution(in_channels, out_channels, adj_matrix)
        self.t_conv = TemporalConvolution(out_channels, out_channels, kernel_size=t_kernel_size, stride=stride)
        self.relu = nn.ReLU()
        
        if not residual:
            self.residual = None
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res_connection = x
        x = self.relu(self.gcn(x))
        x = self.relu(self.t_conv(x))
        
        if self.residual is not None:
            b, c, t, v, m = res_connection.size()
            
            # Safe folding for residual link
            res_safe = res_connection.permute(0, 4, 1, 2, 3).contiguous()
            res_reshaped = res_safe.view(b * m, c, t, v)
            
            res_output = self.residual(res_reshaped) 
            
            # Safe unfolding for residual link
            res_output = res_output.view(b, m, res_output.size(1), res_output.size(2), res_output.size(3))
            res_output = res_output.permute(0, 2, 3, 4, 1).contiguous()
            
            x = x + res_output
            
        return self.relu(x)
import torch
import torch.nn as nn
import numpy as np

class SpatialGraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, norm_adj_matrix, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.register_buffer('adj', torch.from_numpy(norm_adj_matrix).float())
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        b, c, t, v, m = x.size()
        x_reshaped = x.permute(0, 2, 3, 4, 1).contiguous() 
        x_reshaped = x_reshaped.view(b * t * m, v, c) 
        
        support = torch.matmul(x_reshaped, self.adj) 
        output = torch.matmul(support, self.weight) 
        
        if self.bias is not None:
            output += self.bias
            
        output = output.view(b, t, m, v, self.out_channels)
        output = output.permute(0, 4, 1, 3, 2).contiguous() 
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
        x_reshaped = x.view(b * m, c, t, v) 
        output = self.t_conv(x_reshaped) 
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
            b, c, t, v, m = x.size()
            x_res = res_connection.view(b * m, c, t, v)
            output = self.residual(x_res) 
            x = x + output.view(b, m, x.size(1), x.size(2), x.size(3)).permute(0, 2, 3, 4, 1)
        return self.relu(x)
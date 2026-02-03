import torch.nn as nn 
import torch  

class basic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.feat_dim = args.feat_dim 
        self.dropout = args.dropout 
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.feature_dim = self.feat_dim
        self.block1 = nn.Sequential(
            self.conv_layer(self.feat_dim, 3),
            self.conv_layer(self.feat_dim),
            self.conv_layer(self.feat_dim),
        )
        self.block2 = nn.Sequential(
            self.conv_layer(self.feat_dim),
            self.conv_layer(self.feat_dim),
            self.conv_layer(self.feat_dim),
        )
        self.block3 = nn.Sequential(
            self.conv_layer(self.feat_dim),
            self.conv_layer(self.feat_dim),
            self.conv_layer(self.feat_dim)
            )
        self.transit_layer = self.pool_layer()
        
    def forward(self, x):
        x = self.block1(x)
        self.noise(self.transit_layer(x))
        
        x = self.block2(x)
        self.noise(self.transit_layer(x))
        
        x = self.block3(x)
        
        x = x.mean(dim= (-2, -1))
        return x     
        
    def conv_layer(self, out, inp=None):
        inp = out if inp is None else inp 
        return nn.Sequential(
            nn.Conv2d(inp, out, 3, 1, 1),
            nn.BatchNorm2d(out),
            self.activation
        )
    def pool_layer(self):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride= 2),
            nn.Dropout(self.dropout)
        )    
    def noise(self, x, std=1.0):
        eps = torch.randn(x.size()) * std 
        out = x 
        if self.training:
            out += eps.to(x.device)
        return out 
    
    
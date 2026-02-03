import torch.nn as nn 
class FF(nn.Module):
    def __init__(self, args, dim_input, dim_hidden, dim_output, dropout=0.0):
        super().__init__()
        self.residual = args.ff_residual_connection
        self.num_layers = args.ff_layers
        self.use_ln = args.ff_layer_norm
        self.activation = args.ff_activation

        self.blocks = nn.ModuleList()
        for l in range(self.num_layers):
            in_dim = dim_input if l == 0 else dim_hidden
            block = []
            if self.use_ln:
                block.append(nn.LayerNorm(in_dim))
            block.append(nn.Linear(in_dim, dim_hidden))
            block.append({'tanh': nn.Tanh(), 'relu': nn.ReLU()}[self.activation])
            block.append(nn.Dropout(dropout))
            self.blocks.append(nn.Sequential(*block))
            
        self.out = nn.Linear(dim_input if self.num_layers < 1 else dim_hidden, dim_output)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x) if self.residual else block(x) 
        return self.out(x)

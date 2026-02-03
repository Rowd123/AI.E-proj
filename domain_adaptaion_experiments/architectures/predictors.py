import torch.nn as nn 

class DomainPredictor(nn.Module):
    def __init__(self, args, input_dim, num_domains):
        super().__init__()
        self.latent_dim = args.latent_dim_d
        self.activation = nn.ReLU(inplace=True)
        self.block = nn.Sequential(nn.Linear(input_dim, self.latent_dim),
                                   self.activation,
                                   nn.Linear(self.latent_dim, num_domains))
    def forward(self, x):
        x = self.block(x)
        return x 

class LabelPredictor(nn.Module):
    def __init__(self, args, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        x = self.fc(x)
        return x 
    
import torch.nn as nn
from .utils import FF


class CLUB(nn.Module):  
    def __init__(self, args, zc_dim, zd_dim):
        super().__init__()
        self.use_tanh = args.use_tanh
        self.p_mu = FF(args, zc_dim, zc_dim, zd_dim)
        self.p_logvar = FF(args, zc_dim, zc_dim, zd_dim)

    def get_mu_logvar(self, z_c):
        mu = self.p_mu(z_c)
        logvar = self.p_logvar(z_c)
        if self.use_tanh:
            logvar = logvar.tanh()
        return mu, logvar

    def forward(self, z_c, z_d):
        mu, logvar = self.get_mu_logvar(z_c)

        positive = - (mu - z_d) ** 2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1) 
        z_d_1 = z_d.unsqueeze(0) 

        negative = - ((z_d_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()
        mi = (positive.sum(-1) - negative.sum(-1)).mean()
        return mi, 0., 0.

    def learning_loss(self, z_c, z_d):
        mu, logvar = self.get_mu_logvar(z_c)
        return -(-(mu - z_d) ** 2 / logvar.exp() - logvar).sum(1).mean(0)

    def I(self, *args, **kwargs):
        return self.forward(*args[:2], **kwargs)[0]
import numpy as np
import math
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import seaborn as sns
from tensorboardX import SummaryWriter
from mi_utils import *
import random 
import pandas as pd

sns.set_style("white")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 4, 'lines.markersize': 10})


def control_weights(model):
    def init_weights(m):
        if hasattr(m, 'weight') and hasattr(m.weight, 'uniform_') and True:
            torch.nn.init.uniform_(m.weight, a=-0.01, b=0.01)

    model.apply(init_weights)
    
def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128, to_cuda=False, cubic=False):
    """Generate samples from a correlated Gaussian distribution."""
    mean = [0, 0]
    cov = [[1.0, rho], [rho, 1.0]]
    x, y = np.random.multivariate_normal(mean, cov, batch_size * dim).T

    x = x.reshape(-1, dim)
    y = y.reshape(-1, dim)

    if cubic:
        y = y ** 3

    if to_cuda:
        x = torch.from_numpy(x).float().cuda()
        y = torch.from_numpy(y).float().cuda()
    return x, y


def rho_to_mi(rho, dim):
    result = -dim / 2 * np.log(1 - rho ** 2)
    return result


def mi_to_rho(mi, dim):
    result = np.sqrt(1 - np.exp(-2 * mi / dim))
    return result


import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(cubic=True):

    for seed in [1, 2 ,3, 4, 5, 6, 7, 8]:
        set_seed(seed)
        lambda_ = 2

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        suffix = '9.07_{}_{}_{}'.format(cubic, lambda_, seed)

        sample_dim = 20
        batch_size = 128
        hidden_size = 15
        learning_rate = 0.001
        training_steps = 5000
        model_list = ["TUBA", "KNIFE"]  

        mi_list = [2.0, 4.0, 6.0, 8.0, 10.0] 

        total_steps = training_steps * len(mi_list)


        mi_results = dict()
        for model_name in tqdm(model_list, 'Models'):
            if model_name == 'Kernel_F':
                model = MIKernelEstimator(device, sample_dim // 2, sample_dim).to(device)
            elif model_name == 'KNIFE':
                model = MIKernelEstimator(device, batch_size // 6, sample_dim, sample_dim, use_joint=True).to(device)
            elif model_name == 'DOE':
                model = eval(model_name)(sample_dim, sample_dim).to(device)
            else:
                model = eval(model_name)(sample_dim, sample_dim, hidden_size).to(device)
            optimizer = torch.optim.Adam(model.parameters(), learning_rate)

            mi_est_values = []

            start_time = time.time()
            for mi_value in tqdm(mi_list, 'MI'):
                rho = mi_to_rho(mi_value, sample_dim)

                for step in tqdm(range(training_steps), 'Training Loop', position=0, leave=True):
                    batch_x, batch_y = sample_correlated_gaussian(rho, dim=sample_dim, batch_size=batch_size,
                                                                  to_cuda=torch.cuda.is_available(), cubic=cubic)
                    batch_x = torch.tensor(batch_x).float().to(device)
                    batch_y = torch.tensor(batch_y).float().to(device)
                    model.eval()
                    mi_est_values.append(model(batch_x, batch_y).item())

                    model.train()

                    model_loss = model.learning_loss(batch_x, batch_y)

                    optimizer.zero_grad()
                    model_loss.backward()
                    optimizer.step()

                    del batch_x, batch_y
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                print("finish training for %s with true MI value = %f" % (model.__class__.__name__, mi_value))

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            end_time = time.time()
            time_cost = end_time - start_time
            print("model %s average time cost is %f s" % (model_name, time_cost / total_steps))
            mi_results[model_name] = mi_est_values

        import seaborn as sns
        import pandas as pd

        colors = sns.color_palette()

        EMA_SPAN = 200

        ncols = len(model_list)
        nrows = 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.4 * nrows))
        axs = np.ravel(axs)

        xaxis = np.array(list(range(total_steps)))
        yaxis_mi = np.repeat(mi_list, training_steps)

        for i, model_name in enumerate(model_list):
            plt.sca(axs[i])
            p1 = plt.plot(mi_results[model_name], alpha=0.4, color=colors[0])[0]  
            plt.locator_params(axis='y', nbins=5)
            plt.locator_params(axis='x', nbins=4)
            mis_smooth = pd.Series(mi_results[model_name]).ewm(span=EMA_SPAN).mean()

            if i == 0:
                plt.plot(mis_smooth, c=p1.get_color(), label='$\\hat{I}$')
                plt.plot(yaxis_mi, color='k', label='True')
                plt.xlabel('Steps', fontsize=25)
                plt.ylabel('MI', fontsize=25)
                plt.legend(loc='upper left', prop={'size': 15})
            else:
                plt.plot(mis_smooth, c=p1.get_color())
                plt.yticks([])
                plt.plot(yaxis_mi, color='k')
                plt.xlabel('Steps', fontsize=25)

            plt.ylim(0, 15.5)
            plt.xlim(0, total_steps)
            plt.title(model_name, fontsize=35)
            import matplotlib.ticker as ticker

            ax = plt.gca()
            ax.xaxis.set_major_formatter(ticker.EngFormatter())
            plt.xticks(horizontalalignment="right")
            # plt.subplots_adjust( )

        plt.gcf().tight_layout()
        plt.savefig('mi_est_Gaussian_{}_copy.pdf'.format(suffix), bbox_inches=None)
        # plt.show()

        print('Second part')

        bias_dict = dict()
        var_dict = dict()
        mse_dict = dict()
        for i, model_name in tqdm(enumerate(model_list)):
            bias_list = []
            var_list = []
            mse_list = []
            for j in range(len(mi_list)):
                mi_est_values = mi_results[model_name][training_steps * (j + 1) - 500:training_steps * (j + 1)]
                est_mean = np.mean(mi_est_values)
                bias_list.append(np.abs(mi_list[j] - est_mean))
                var_list.append(np.var(mi_est_values))
                mse_list.append(bias_list[j] ** 2 + var_list[j])
            bias_dict[model_name] = bias_list
            var_dict[model_name] = var_list
            mse_dict[model_name] = mse_list



        plt.style.use('default') 

        colors = list(plt.rcParams['axes.prop_cycle'])
        col_idx = [2, 4, 5, 1, 3, 0, 6, 7]

        ncols = 1
        nrows = 3
        fig, axs = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3. * nrows))
        axs = np.ravel(axs)

        for i, model_name in enumerate(model_list):
            plt.sca(axs[0])
            plt.plot(mi_list, bias_dict[model_name], label=model_name, marker='d', color=colors[col_idx[i]]["color"])

            plt.sca(axs[1])
            plt.plot(mi_list, var_dict[model_name], label=model_name, marker='d', color=colors[col_idx[i]]["color"])

            plt.sca(axs[2])
            plt.plot(mi_list, mse_dict[model_name], label=model_name, marker='d', color=colors[col_idx[i]]["color"])

        ylabels = ['Bias', 'Variance', 'MSE']
        for i in range(3):
            plt.sca(axs[i])
            plt.ylabel(ylabels[i], fontsize=15)

            if i == 0:
                if cubic:
                    plt.title('Cubic', fontsize=17)
                else:
                    plt.title('Gaussian', fontsize=17)
            if i == 1:
                plt.yscale('log')
            if i == 2:
                plt.legend(loc='upper left', prop={'size': 12})
                plt.xlabel('MI Values', fontsize=15)

        plt.gcf().tight_layout()
        plt.savefig('bias_variance_Gaussian_{}.pdf'.format(suffix), bbox_inches='tight')


if __name__ == '__main__':
    main()
        

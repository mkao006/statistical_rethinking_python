import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist

from pyro.nn import PyroModule
from pyro.nn import PyroSample
from pyro.nn import PyroParam
from pyro.infer import SVI, Trace_ELBO

from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc import MCMC
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.infer.autoguide import AutoDiagonalNormal

import pandas as pd
from pandas.io.formats.format import format_percentiles

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns


class BayesianMCMCLinearModel():
    def __init__(self, warmup_steps=100, num_samples=1000):
        self.warmup_steps = warmup_steps
        self.num_samples = num_samples

    def model(self, X, y):
        input_size = X.shape[0]

        # sample
        bias = pyro.sample('bias', self.bias_prior)
        weights = pyro.sample('weights', self.weights_prior)
        sigma = pyro.sample('sigma', self.sigma_prior)
        prior_mean = bias + X @ weights
        self.obs_distribution = dist.Normal(prior_mean, sigma)
        with pyro.plate('data', input_size):
            return pyro.sample('obs', self.obs_distribution, obs=y)

    def fit(self, X, y):
        pyro.clear_param_store()
        self.X = X
        self.y = y

        self._create_prior(X, y)
        self.kernel = NUTS(self.model)
        self.mcmc = MCMC(self.kernel,
                         warmup_steps=self.warmup_steps,
                         num_samples=self.num_samples)
        self.mcmc.run(X, y)
        self._posterior_sample_df()

    def generate_posterior_samples(self,
                                   n_samples=1000):
        return self.mcmc.get_samples(n_samples)

    def posterior_summary(self, q=[0.05, 0.95], plot=False):
        if self.posterior_df is None:
            self._posterior_sample_df()
        summary = self.posterior_df.describe(percentiles=q).T
        if not plot:
            return summary
        else:
            sns.scatterplot(x=summary['mean'], y=summary.index)
            for i, var in enumerate(summary.index):
                sns.lineplot(x=summary.loc[var, format_percentiles(q)],
                             y=[var, var], color='k')
            plt.xlabel('')
            plt.show()

    def plot_joint_posterior(self):
        g = sns.pairplot(self.posterior_df, diag_kind='kde', corner=True)
        g.map_lower(sns.kdeplot, levels=4, color='.2')
        plt.show()

    def plot_counterfactual(self):
        # counter factual plots
        pass

    def plot_dist(self):
        # need to determine the total number of param
        param_num = self.posterior_df.shape[1]

        # plot bias
        plt.subplot(param_num, 1, 1)
        prior_sample = pd.DataFrame({'value': lm.bias_prior.rsample((1000, )), 'type': 'prior'})
        posterior_sample = pd.DataFrame({'value': lm.posterior_df['bias'], 'type': 'posterior'})
        plot_df = pd.concat([prior_sample, posterior_sample])
        ax = sns.histplot(data=plot_df, x='value', hue='type', kde=True)
        ax.set(xlabel='bias', ylabel = '')

        # plot weights
        w = lm.weights_prior.rsample((1000, ))
        for i in range(w.shape[1]):
            plt.subplot(param_num, 1, i + 2)
            prior_sample = pd.DataFrame({'value': w[:, i], 'type': 'prior'})
            posterior_sample = pd.DataFrame({'value': lm.posterior_df[f'weights_{i + 1}'],
                                             'type': 'posterior'})
            plot_df = pd.concat([prior_sample, posterior_sample])
            ax = sns.histplot(data=plot_df, x='value', hue='type', kde=True)
            ax.set(xlabel=f'weights_{i + 1}', ylabel = '')

        # plot sigma
        plt.subplot(param_num, 1, param_num)
        prior_sample = pd.DataFrame({'value': lm.sigma_prior.rsample((1000, )), 'type': 'prior'})
        posterior_sample = pd.DataFrame({'value': lm.posterior_df['sigma'], 'type': 'posterior'})
        plot_df = pd.concat([prior_sample, posterior_sample])
        ax = sns.histplot(data=plot_df, x='value', hue='type', kde=True)
        ax.set(xlabel='sigma', ylabel = '')

        plt.show()

    def plot_predicted(self):
        ps = self.posterior_summary()
        expected_bias = ps['mean']['bias']
        expected_weights = ps['mean'][ps['mean'].index.str.startswith('weights')]
        y_pred = expected_bias + self.X @ expected_weights
        ax = sns.scatterplot(self.y, y_pred)
        ax.set(xlabel='Observed', ylabel='predicted')
        ax.plot([self.y.min(), self.y.max()], [y_pred.min(), y_pred.max()], ls="--", c=".3")
        plt.show()

    def _create_prior(self, X, y):
        input_size = X.shape[0]
        self.input_dim = X.shape[1]
        
        w_mu_init = torch.zeros(self.input_dim)
        w_sigma_init = torch.eye(self.input_dim)

        self.bias_prior = dist.Normal(0, 10)
        self.weights_prior = dist.MultivariateNormal(w_mu_init, w_sigma_init)
        self.sigma_prior = dist.InverseGamma(10, 50)

    def _posterior_sample_df(self):
        posterior_sample = self.generate_posterior_samples(n_samples=1000)
        result = {}
        for k, v in posterior_sample.items():
            if len(v.shape) == 1:
                result[k] = v.numpy()
            else:
                for i in range(v.shape[1]):
                    result[f'{k}_{i + 1}'] = v[:, i].numpy()
        self.posterior_df = pd.DataFrame(result)
        

class BayesianSVILinearModel():
    def __init__(self, lr=0.01, n_iter=1000):
        super().__init__()
        self.optimiser = pyro.optim.Adam({'lr': lr})
        self.n_iter = n_iter
        
        pyro.clear_param_store()
                
    def model(self, X, y):
        input_size = X.shape[0]
        input_dim = X.shape[1]


        # initialise
        w_mu_init = torch.zeros(input_dim)
        w_sigma_init = torch.eye(input_dim)
        self.sigma_concentration = pyro.param('sigma_concentration',
                                              torch.tensor(10),
                                              constraint=constraints.positive)
        self.sigma_rate = pyro.param('sigma_rate',
                                     torch.tensor(50),
                                     constraint=constraints.positive)

        # priors
        self.bias = pyro.sample('bias', dist.Normal(0, 10))
        self.weights = pyro.sample('weights', dist.MultivariateNormal(w_mu_init, w_sigma_init))

        self.sigma = pyro.sample('sigma',
                                 dist.InverseGamma(concentration=self.sigma_concentration,
                                                   rate=self.sigma_rate))

        # expected and obs
        prior_mean = self.bias + X @ self.weights
        with pyro.plate('data', input_size):
            return pyro.sample('obs', dist.Normal(prior_mean, self.sigma), obs=y)

    def fit(self, X, y):

        # create the guide
        self.guide = self._create_guide(self.model)

        # initialise the svi
        self.svi = SVI(self.model, self.guide, self.optimiser, Trace_ELBO())

        # train the model
        self._train(X, y)

    def _train(self, X, y):
        self.losses = []
        for i in range(self.n_iter):
            loss = self.svi.step(X, y)
            self.losses.append(loss)
            if i % 50 == 0:
                print('[Iteration %04d] loss: %.4f' % (i, loss))            

    def _create_guide(self, X):
        return AutoMultivariateNormal(self.model, init_scale=0.2)

    def plot_losses(self):
        sns.lineplot(x=range(self.n_iter), y=self.losses)
        plt.show()

from functools import partial
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


class BayesianLinearModel():
    '''Object to hold the data and the model of a Bayesian linear model.

    '''
    def __init__(self, X, y, prior):
        self.X = X
        self.y = y
        # we can make prior optional here.
        self.prior = prior
        self.model = self._create_linear_model()

    def _create_linear_model(self):
        ''' Create a linear model from the prior specified.
        '''
        input_size = X.shape[0]

        def _model_(self):
            bias = pyro.sample('bias', self.prior['bias'])
            weights = pyro.sample('weights', self.prior['weights'])
            sigma = pyro.sample('sigma', self.prior['sigma'])
            prior_mean = bias + self.X @ weights
            self.obs_distribution = dist.Normal(prior_mean, sigma)
            with pyro.plate('data', input_size):
                return pyro.sample('obs', self.obs_distribution, obs=self.y)
        return partial(_model_, self)

    def _generate_reference_prior(self):
        '''create a reference prior in the absence of a model.

        '''
        pass


class BayesianMCMCLinearModel(BayesianLinearModel):
    def __init__(self, X, y, prior, warmup_steps=100, num_samples=1000):
        super().__init__(X, y, prior)
        self.warmup_steps = warmup_steps
        self.num_samples = num_samples


    def fit(self):
        pyro.clear_param_store()
        self.kernel = NUTS(self.model)
        self.mcmc = MCMC(self.kernel,
                         warmup_steps=self.warmup_steps,
                         num_samples=self.num_samples)
        self.mcmc.run()
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

    def _plot_prior_posterior(self, prior_sample, posterior_sample, label):
        plot_df = pd.concat([pd.DataFrame({'value': prior_sample, 'type': 'prior'}),
                             pd.DataFrame({'value': posterior_sample, 'type': 'posterior'})])
        ax = sns.histplot(data=plot_df, x='value', hue='type', kde=True)
        ax.set(xlabel = '', ylabel=label)


    def plot_prior_posterior(self, sample_size=1000):
        # need to determine the total number of param
        param_num = self.posterior_df.shape[1]

        # plot bias
        plt.subplot(param_num, 1, 1)
        self._plot_prior_posterior(prior_sample=self.prior['bias'].rsample((sample_size, )),
                                   posterior_sample=self.posterior_df['bias'],
                                   label='bias')

        # plot weights
        w = self.prior['weights'].rsample((1000, ))
        for i in range(w.shape[1]):
            plt.subplot(param_num, 1, i + 2)
            self._plot_prior_posterior(prior_sample=w[:, i],
                                       posterior_sample=self.posterior_df[f'weights_{i + 1}'],
                                       label=f'weights_{i + 1}')

        # plot sigma
        plt.subplot(param_num, 1, param_num)
        self._plot_prior_posterior(self.prior['sigma'].rsample((sample_size, )),
                                   self.posterior_df['sigma'],
                                   label='sigma')
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

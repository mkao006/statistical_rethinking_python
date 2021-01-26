import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoLaplaceApproximation
from pyro.infer import SVI
from pyro.infer import Trace_ELBO
from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc import MCMC

class BinomialGridApproximator():
    def __init__(self, X, N, grid_size, prior_type):
        self.X = X
        self.N = N
        self.grid_size = grid_size

        self.param_grid = np.linspace(0, 1, self.grid_size)

        self.prior_type = prior_type
        self.prior = self._generate_prior()
        self.likelihood = binom.pmf(X, N, self.param_grid)
        self.posterior = self._compute_posterior()

    def _generate_prior(self):
        if self.prior_type == 'uniform':
            return np.repeat(1, self.grid_size)
        elif self.prior_type == 'positive':
            return (self.param_grid > 0.5).astype(int)
        elif self.prior_type == 'laplace':
            return np.exp(-0.5 * abs(self.param_grid - 0.5))
        else:
            raise NotImplementedError

    def plot(self):

        # plot prior
        plt.subplot(3, 1, 1)        
        plt.plot(self.param_grid, self.prior, '-o')
        plt.title('Prior distribution')

        # plot likelihood
        plt.subplot(3, 1, 2)
        plt.plot(self.param_grid, self.likelihood, '-o')
        plt.title('Likelihood')
        
        # plot posterior
        plt.subplot(3, 1, 3)
        plt.plot(self.param_grid, self.posterior, '-o')
        plt.title('Posteriod Distribution')

    def _compute_posterior(self):
        posterior = self.prior * self.likelihood
        normalised_posterior = posterior/posterior.sum()
        return normalised_posterior

    def sample_posterior(self, sample_size):
        return np.random.choice(self.param_grid, sample_size, p=self.posterior)
        


class BinomialQuadraticApproximator():
    def __init__(self, X, N, n_steps, learning_rate, prior_type, infer_type):
        self.X = X
        self.N = N
        self.n_steps = n_steps
        self.prior_type = prior_type
        self.infer_type = infer_type

        self.optimiser = pyro.optim.Adam({'lr': learning_rate})
        self.map_guide = AutoLaplaceApproximation(self.model)

    def plot(self):
        plt.subplot(3, 1, 1)
        plt.plot(self.losses)
        plt.title('losses')

        plt.subplot(3, 1, 2)
        plt.plot(self._posterior_approximate_mean)
        plt.title('Posterior Mean')

        plt.subplot(3, 1, 3)
        plt.plot(self._posterior_approximate_scale)
        plt.title('Posterior Scale (Variance)')

    def train(self, **kwargs):
        pyro.clear_param_store()
        if self.infer_type == 'svi':
            self._svi_trainer(**kwargs)
        elif self.infer_type == 'mcmc':
            self._mcmc_trainer(**kwargs)
        

    def _svi_trainer(self):
        svi = SVI(self.model, self.map_guide, self.optimiser, Trace_ELBO())
        self.losses = []
        self._posterior_approximate_mean = []
        self._posterior_approximate_scale = []
        for step in range(self.n_steps):
            loss = svi.step(self.X)
            self.losses.append(loss)
            quadratic_approximation = self.map_guide.laplace_approximation(self.X).get_posterior()
            self._posterior_approximate_mean.append(quadratic_approximation.loc.item())
            self._posterior_approximate_scale.append(quadratic_approximation.scale_tril.item())

            if step % 50 == 0:
                print('[iter {}]  loss: {:.4f}'.format(step, loss))

    def _mcmc_trainer(self, step_size=0.1, num_samples=1000, warmup_steps=100):
        mcmc_kernel = NUTS(self.model, step_size=step_size)
        self.mcmc = MCMC(mcmc_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
        self.mcmc.run(self.X)
        

    def model(self, data):
        if self.prior_type == 'uniform':
            p = pyro.sample('p', dist.Uniform(0, 1))
        elif self.prior_type == 'beta':
            p = pyro.sample('p', dist.Beta(10.0, 10.0))
        return pyro.sample('obs', dist.Binomial(total_count=self.N, probs=p), obs=self.X)

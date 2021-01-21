import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

class BinomialGridApproximator():
    def __init__(self, X, N, sample_size, prior_type):
        self.X = X
        self.N = N
        self.sample_size = sample_size

        self.param_grid = np.linspace(0, 1, self.sample_size)

        self.prior_type = prior_type
        self.prior = self._generate_prior()
        self.likelihood = binom.pmf(X, N, self.param_grid)
        self.posterior = self._compute_posterior()

    def _generate_prior(self):
        if self.prior_type == 'uniform':
            return np.repeat(1, self.sample_size)
        elif self.prior_type == 'positive':
            return (self.param_grid > 0.5).astype(int)
        elif self.prior_type == 'exponential':
            return np.exp(-5 * abs(self.param_grid - 0.5))
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
        return posterior/posterior.sum()

class BinomialQuadraticApproximator():
    def __init__(self, X, N):
        self.X = X
        self.N = N

        self.prior = self._generate_prior()
        self.likelihood = binom.pmf(X, N, self.param_grid)
        self.posterior = self._compute_posterior()

    def plot(self):
        pass

    def _compute_posterior(self):
        # estimate the mode using MAP
        pass

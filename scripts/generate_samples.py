from notebooks.variable_selection.MosesEstimator import *
from notebooks.variable_selection.mixed_hmc_jax import *

import numpy as np
import jax
import jax.numpy as jnp
from scripts.samplers import *

def draw_samples_mh_mi(X, y, J, n_chains, num_iter, num_samples, eta, mu, B):

    _, _, samples = metropolis(J, num_iter, bmm_energy, X, y, n_chains, num_samples, eta, mu)

    feats_mi, _ = rank_by_mi(samples, J, X, y, eta, mu, B)

    return feats_mi

def draw_samples_hmc(key, X, y, J, sigma, n_chains, n_samples, n_warm_up, eta, mu, rw=False):

    kernel = MixedHMC(HMC(model), num_discrete_updates=X.shape[0], random_walk=rw)
    mcmc = MCMC(kernel, num_warmup=n_warm_up, num_samples=n_samples, num_chains=n_chains)
    mcmc.run(key, X, y, sigma, J, eta, mu)

    gamma_samples = jax.device_get(mcmc.get_samples()["gamma"])

    return rank_hmc_feats_rand(gamma_samples)


def draw_samples_hmc_mi(key, X, y, J, sigma, n_chains, n_samples, n_warm_up, eta, mu, B, rw=False):
    kernel = MixedHMC(HMC(model), num_discrete_updates=X.shape[0], random_walk=rw)
    mcmc = MCMC(kernel, num_warmup=n_warm_up, num_samples=n_samples, num_chains=n_chains)
    mcmc.run(key, X, y, sigma*np.identity(X.shape[1]), J, eta, mu)

    gamma_samples = jax.device_get(mcmc.get_samples()["gamma"])
    beta_samples = jax.device_get(mcmc.get_samples()["beta"])

    feats, _ = rank_hmc_feats_mi(gamma_samples, beta_samples, X, y, J, sigma, eta, mu, B)

    return feats

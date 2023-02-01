#!/usr/bin/env python3

import argparse
import os

# set the following jax env variables before importing jax

os.environ["XLA_FLAGS"] = '--xla_gpu_autotune_level=0'
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from typing import Callable, NamedTuple
from blackjax.mcmc.diffusion import generate_gaussian_noise
from blackjax.mcmc.mala import MALAState
from blackjax.types import PRNGKey, PyTree
import time
import joblib
import datetime
import itertools
from sklearn.model_selection import train_test_split
import pandas as pd
tfd = tfp.distributions


class MixedMALAState(NamedTuple):
    """Holds info about the discrete and the continuous r.vs in the mixed support"""

    discrete_position: PyTree
    contin_position: PyTree

    disc_logprob: float
    contin_logprob: float

    discrete_logprob_grad: PyTree
    contin_logprob_grad: PyTree

    disc_step_size: float
    contin_step_size: float




EPS = 1e-10

def build_network(X, net_intr, net_intr_rev):
    p = X.shape[1]
    J = np.zeros((p, p))
    cols = X.columns
    intrs = []
    intrs_rev = []
    for i, g1 in enumerate(cols):
        try:
            g_intrs = list(net_intr[g1])
            for g2 in g_intrs:
                if (g2, g1) not in intrs_rev: # check if we haven't encountered the reverse interaction
                    j = cols.get_loc(g2)
                    J[i, j] = 1.0
                    J[j, i] = 1.0
                    intrs.append((g1, g2))
        except KeyError:
            continue

        # Check the reverse direction
        try:
            g_intrs_rev = list(net_intr_rev[g1])
            for g2 in g_intrs_rev:
                if (g1, g2) not in intrs:
                    j = cols.get_loc(g2)
                    J[i, j] = 1.0
                    J[j, i] = 1.0
                    intrs_rev.append((g2, g1))

        except KeyError:
            continue


    return J

def diff_fn(state, step_size):
    theta = jax.tree_util.tree_map(lambda x, g: -0.5 * (g) * (2. * x - 1) - (1. / (2. * step_size)),
                                   state.position, state.logprob_grad)

    return jax.nn.sigmoid(theta)


def take_discrete_step(rng_key: PRNGKey, disc_state: MALAState, contin_state: MALAState,
                       logprob_fn: Callable, disc_grad_fn: Callable,
                       step_size: float) -> MALAState:
    _, key_rmh, key_accept = jax.random.split(rng_key, 3)
    # key_integrator, key_rmh = jax.random.split(rng_key)
    theta_cur = disc_state.position

    u = jax.random.uniform(key_rmh, shape=disc_state.position.shape)
    p_curr = diff_fn(disc_state, step_size)
    ind = jnp.array(u < p_curr)
    pos_new = (1. - theta_cur) * ind + theta_cur * (1. - ind)

    logprob_new = logprob_fn(pos_new, contin_state.position)
    logprob_grad_new = disc_grad_fn(pos_new, contin_state.position)
    new_state = MALAState(pos_new, logprob_new, logprob_grad_new)  # No metropolis update just accept the move

    return new_state


def take_contin_step(rng_key: PRNGKey, disc_state: MALAState, contin_state: MALAState,
                     logprob_fn: Callable, contin_grad_fn: Callable,
                     step_size: float) -> MALAState:
    key_integrator, key_rmh = jax.random.split(rng_key)
    noise = generate_gaussian_noise(key_integrator, contin_state.position)
    new_position = jax.tree_util.tree_map(
        lambda p, g, n: p + step_size * g + jnp.sqrt(2 * step_size) * n,
        contin_state.position,
        contin_state.logprob_grad,
        noise,
    )

    logprob_new = logprob_fn(disc_state.position, new_position)
    logprob_grad_new = contin_grad_fn(disc_state.position, new_position)
    new_state = MALAState(new_position, logprob_new, logprob_grad_new)

    return new_state


def one_step(
        rng_key: PRNGKey, state: MixedMALAState,
        discrete_logprob_fn: Callable, contin_logprob_fn: Callable,
        discrete_step_size: float, contin_step_size: float
) -> MixedMALAState:
    disc_grad_fn = jax.grad(discrete_logprob_fn)
    contin_grad_fn = jax.grad(contin_logprob_fn, argnums=1)
    # Evolve each variable in tandem and combine the results

    disc_state = MALAState(state.discrete_position, state.disc_logprob, state.discrete_logprob_grad)
    contin_state = MALAState(state.contin_position, state.contin_logprob, state.contin_logprob_grad)
    # print(f"disc pos: {disc_state.position}, contin pos: {contin_state.position}")
    # Take a step for the discrete variable - sample from p(discrete | contin)
    new_disc_state = take_discrete_step(rng_key, disc_state, contin_state,
                                        discrete_logprob_fn, disc_grad_fn, discrete_step_size)
    # Take a step for the contin variable - sample from p(contin | new_discrete)
    new_contin_state = take_contin_step(rng_key, new_disc_state, contin_state,
                                        contin_logprob_fn, contin_grad_fn, contin_step_size)

    new_state = MixedMALAState(new_disc_state.position, new_contin_state.position,
                               new_disc_state.logprob, new_contin_state.logprob,
                               new_disc_state.logprob_grad, new_contin_state.logprob_grad,
                               discrete_step_size, contin_step_size)

    return new_state

def init(disc_position: PyTree,contin_position: PyTree,
         disc_logprob_fn: Callable, contin_logprob_fn: Callable,
         init_disc_step: float, init_contin_step: float) -> MixedMALAState:

    disc_logprob, disc_grad_logprob = jax.value_and_grad(disc_logprob_fn)(disc_position, contin_position)
    contin_logprob, contin_grad_logprob = jax.value_and_grad(contin_logprob_fn, argnums=1)(disc_position, contin_position)

    return MixedMALAState(disc_position, contin_position,
                          disc_logprob, contin_logprob,
                          disc_grad_logprob, contin_grad_logprob,
                          init_disc_step, init_contin_step)
#%%
def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states

def inference_loop_multiple_chains(rng_key, kernel, initial_state, num_samples, num_chains):

    @jax.jit
    def one_step(state, step_key):
        subkeys = jax.random.split(step_key, num_chains)
        state = jax.vmap(kernel)(subkeys, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states
#%%
def gamma_energy(theta, J, eta, mu):
    xg = theta.T @ J
    xgx = xg @ theta
    return eta*xgx - mu*jnp.sum(theta)

def generate_disc_logprob_fn(X, y, J, mu, eta):

    def discrete_logprob_fn(gamma, beta):
        # beta = pos["beta"]
        X_gamma = (X @ jnp.diag(gamma))
        ising_logp = gamma_energy(gamma, J, eta, mu)
        ll_dist = tfd.Bernoulli(logits=(X_gamma @ beta))
        log_ll = jnp.sum(ll_dist.log_prob(y), axis=0)

        # print(f"gamma logp: {ising_logp}, log_ll: {log_ll}")

        return ising_logp + log_ll

    return discrete_logprob_fn


def generate_contin_logprob_fn(X, y, tau, c):
    n, p = X.shape
    cov = X.T @ X
    R = np.identity(p)
    v, l = 1., 1.

    def contin_logprob_fn(gamma, beta):
        # beta = pos["beta"]

        D = (gamma*c*tau) + (1 - gamma)*(tau)
        # D_inv = jnp.linalg.inv(jnp.diag(D))

        # A = jnp.linalg.inv((1./sigma**2)*cov + (D_inv @ R @ D_inv))
        beta_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(p), scale_diag=D)
        # print(beta_dist.sample(seed=rng_key))
        beta_logp = beta_dist.log_prob(beta)
        X_gamma = (X @ jnp.diag(gamma))
        ll_dist = tfd.Bernoulli(logits=(X_gamma @ beta))
        log_ll = jnp.sum(ll_dist.log_prob(y), axis=0)

        # print(f"beta logp: {beta_logp}, log_ll: {log_ll}")

        return beta_logp + log_ll

    return contin_logprob_fn

def run_gibbs_sampling(seed, data_dir, X_df, y_df,
                       eta, mu, thres,
                       net_intr, net_intr_rev):
    start_time = time.time()
    num_chains = 2
    disc_step_size = 0.1
    contin_step_size = 1e-5
    n_steps = 10000
    tau, c = 0.01, 1000
    burn_in = 0.1

    data_path = f"{data_dir}/exp_data_2"

    key = jax.random.PRNGKey(seed)
    jnp.save(f"{data_path}/rand_test/local/jax_key.npy", key)
    np.random.random(seed)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, shuffle=True, random_state=seed,
                                                        stratify=y_df, test_size=0.3)
    idx_sig = np.load(f"{data_dir}/exp_data_2/npy/idx_sig_s_{seed}.npy")
    X_train, X_test = X_train.iloc[:,idx_sig], X_test.iloc[:,idx_sig]
    np.save(f"{data_path}/rand_test/local/X_train.npy", X_train)
    np.save(f"{data_path}/rand_test/local/X_test.npy", X_test)
    np.save(f"{data_path}/rand_test/local/y_train.npy", y_train)
    np.save(f"{data_path}/rand_test/local/y_test.npy", y_test)
    J = build_network(X_train, net_intr, net_intr_rev)
    p = J.shape[1]
    print(f"dim: {p}")
    np.save(f"{data_path}/rand_test/local/J_mat.npy", J)
    beta_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(p), scale_diag=10 * jnp.ones(p))
    gamma_dist = tfd.Bernoulli(probs=0.5 * jnp.ones(p))

    contin_init_pos = beta_dist.sample(seed=key, sample_shape=(num_chains,))
    disc_init_pos = gamma_dist.sample(seed=key, sample_shape=(num_chains,)) * 1.

    # contin_init_pos = beta_dist.sample(seed=key)
    # disc_init_pos = gamma_dist.sample(seed=key) * 1.

    jnp.save(f"{data_dir}/exp_data_2/rand_test/local/contin_init_pos.npy", contin_init_pos)
    jnp.save(f"{data_dir}/exp_data_2/rand_test/local/disc_init_pos.npy", disc_init_pos)

    X_train_dev, y_train_dev = jax.device_put(X_train.to_numpy()), jax.device_put(y_train.to_numpy())
    disc_logprob = generate_disc_logprob_fn(X_train_dev, y_train_dev, J, mu, eta)
    contin_logprob = generate_contin_logprob_fn(X_train_dev, y_train_dev, tau, c)

    kernel = jax.jit(lambda key, state: one_step(key, state, disc_logprob, contin_logprob, disc_step_size, contin_step_size))

    init_state = jax.vmap(init, in_axes=(0, 0, None, None, None, None))(disc_init_pos, contin_init_pos, disc_logprob, contin_logprob,
                                                                        disc_step_size, contin_step_size)

    # init_state = init(disc_init_pos, contin_init_pos, disc_logprob, contin_logprob,
    #                                  disc_step_size, contin_step_size)

    states = inference_loop_multiple_chains(key, kernel, init_state, num_samples=n_steps, num_chains=num_chains)
    # states = inference_loop(key, kernel, init_state, num_samples=n_steps)
    gamma_samples = states.discrete_position[int(burn_in*n_steps):]
    beta_samples = states.contin_position[int(burn_in*n_steps):]

    np.save(f"{data_path}/rand_test/local/gamma_samples.npy", gamma_samples)
    np.save(f"{data_path}/rand_test/local/beta_samples.npy", beta_samples)

    gamma_samples = gamma_samples.reshape(-1, p)

    # beta_samples = beta_samples.reshape(-1, p)

    gamma_means = jnp.mean(gamma_samples, axis=0)
    idx = jnp.squeeze(jnp.argwhere(gamma_means > thres))
    print(f"---- Inference took {(time.time() - start_time) : .2f} seconds -----")
    return idx

def main(args):
    print("Loading data ....")
    tamox_df = pd.read_csv(f"{args.data_dir}/tamoxBinaryEntrez.csv")
    regnet_df = pd.read_table(f"{args.data_dir}/human.source", sep="\t", header=None, names= ["REGULATOR SYMBOL", "REGULATOR ID", "TARGET SYMBOL", "TARGET ID"])
    net_intr = pd.Series(regnet_df["REGULATOR ID"].values, index=regnet_df["TARGET ID"])
    net_intr_rev = pd.Series(regnet_df["TARGET ID"].values, index=regnet_df["REGULATOR ID"])
    X_df, y_df = tamox_df.iloc[:, 1:], tamox_df["posOutcome"]

    print("Sampling ...")
    idx = run_gibbs_sampling(args.seed, args.data_dir, X_df, y_df, args.eta, args.mu,
                             args.thres, net_intr, net_intr_rev)

    print(f"Num selected: {idx.size}")
    np.save(f"{args.data_dir}/exp_data_2/rand_test/local/idx_sel.npy", idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--mu", type=float)
    parser.add_argument("--thres", type=float)

    args = parser.parse_args()
    main(args)
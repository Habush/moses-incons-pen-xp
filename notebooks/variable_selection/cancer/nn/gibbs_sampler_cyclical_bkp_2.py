## Author Abdulrahman S. Omar <hsamireh@gmail>

from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from blackjax.mcmc.diffusions import generate_gaussian_noise
from blackjax.types import PyTree, PRNGKey

import tree_utils
from nn_util import *

tfd = tfp.distributions

EPS = 1e-5

# Used for RMSProp
class PreconditionState(NamedTuple):
    v: PyTree
    alpha: float

class MixedState(NamedTuple):
    """Holds info about the discrete and the continuous r.vs in the mixed support"""

    count: int
    discrete_position: PyTree
    contin_position: PyTree

    disc_precond: PreconditionState
    contin_precond: PreconditionState


class SGLDState(NamedTuple):
    """Holds infor about a continous r.v - # used when using only the contin variables """
    count: int
    position: PyTree
    precond: PreconditionState


def gamma_energy(theta, J, eta, mu):
    """Log probability of the Ising model - prior over the discrete variables"""
    xg = theta.T @ J
    xgx = xg @ theta
    return eta*xgx - mu*jnp.sum(theta)


def generate_disc_logprior_fn(J, eta, mu):

    def discrete_logprior_fn(gamma):
        log_prior = gamma_energy(gamma, J, eta, mu)
        return log_prior

    return discrete_logprior_fn


# def generate_contin_logprior_fn(sigma):
#
#     def contin_logprior_fn(params):
#         """Computes the Gaussian prior log-density."""
#         weight_decay = 1./sigma
#         n_params = sum([p.size for p in jax.tree_util.tree_leaves(params)])
#         log_prob = -(0.5 * tree_utils.tree_dot(params, params) * weight_decay +
#                      0.5 * n_params * jnp.log(weight_decay / (2 * jnp.pi)))
#         return log_prob
#
#     return contin_logprior_fn

def generate_contin_logprior_fn(sigma):

    def contin_logprior_fn(params):
        """Computes the Gaussian prior log-density."""

        weight_decay = 1./sigma
        n_params = sum([p.size for p in jax.tree_util.tree_leaves(params)])
        log_prob = -(0.5 * tree_utils.tree_dot(params, params) * weight_decay +
                     0.5 * n_params * jnp.log(weight_decay / (2 * jnp.pi)))

        return log_prob

    return contin_logprior_fn

def mixed_loglikelihood_fn(model, params, batch, gamma, loss_fn=cross_entropy_loss):
    batch_size = batch.x.shape[0]
    log_ll = loss_fn(model, batch.x, batch.y, params, gamma)
    return log_ll / batch_size

def loglikelihood_fn(model, params, batch, loss_fn=cross_entropy_loss):
    batch_size = batch.x.shape[0]
    log_ll = loss_fn(model, batch.x, batch.y, params, gamma)
    return log_ll / batch_size

def update_grad(grad_prior, grad_ll, state, data_size, temp):
    v = jax.tree_util.tree_map(lambda v, g: state.alpha*v + (1 - state.alpha)*(g**2),
                               state.v, grad_ll) # Calcuate the Exponentially Moving Weight Average (EWMA)
    grad = jax.tree_util.tree_map(lambda gp, gl: temp*((gp/data_size) + gl), # Acc. Equation 7 of Wenzel et.al 2020 re-scale the prior by 1/n, n - training size
                                  grad_prior, grad_ll)

    return grad, PreconditionState(v, state.alpha)

def generate_discrete_grad_estimator(model, logprior_fn, log_likelihood_fn, data_size, temp=1.0):

    def grad_estimator(params, gamma, state, batch):
        grad_ll = jax.grad(log_likelihood_fn, argnums=3)(model, params, batch, gamma)
        grad_prior = jax.grad(logprior_fn)(gamma)
        return update_grad(grad_prior, grad_ll, state, data_size, temp)

    return grad_estimator



def geneate_mixed_contin_grad_estimator(model, logprior_fn, log_likelihood_fn, data_size, temp=1.0):

    def grad_estimator(params, gamma, state, batch):
        grad_ll = jax.grad(log_likelihood_fn, argnums=1)(model, params, batch, gamma)
        grad_prior = jax.grad(logprior_fn)(params)
        return update_grad(grad_prior, grad_ll, state, data_size, temp)

    return grad_estimator

def generate_sgld_contin_grad_estimator(model, logprior_fn, data_size, temp=1.0):

    def grad_estimator(params, state, batch):
        grad_ll = jax.grad(loglikelihood_fn, argnums=1)(model, params, batch)
        grad_prior = jax.grad(logprior_fn)(params)
        return update_grad(grad_prior, grad_ll, state, data_size, temp)

    return grad_estimator


def take_discrete_step(rng_key: PRNGKey, state: MixedState, disc_grad_fn: Callable,
                       batch: Batch, step_size_fn: Callable) ->Tuple[PyTree, PreconditionState]:

    """SGLD update for the discrete variable. Ref: Zhang et.al 2022 (https://arxiv.org/abs/2206.09914) - Alogrithm 2"""

    _, key_rmh, key_accept = jax.random.split(rng_key, 3)

    disc_pos, contin_pos = state.discrete_position, state.contin_position
    precond = state.disc_precond
    step_size = step_size_fn(state.count)

    u = jax.random.uniform(key_rmh, shape=disc_pos.shape)
    grad, pstate = disc_grad_fn(contin_pos, disc_pos, precond, batch)
    m = jax.tree_util.tree_map(lambda k: 1./(EPS + jnp.sqrt(k)), pstate.v)
    theta = jax.tree_util.tree_map(lambda x, g, i: (-g* (2. * x - 1)) - (1. / (2. * step_size*(i**2))), # Ref Appendix H of Zhang et.al 2022
                                   disc_pos, grad, m)

    # theta = jax.tree_util.tree_map(lambda x, g: -0.5 * (g) * (2. * x - 1) - (1. / (2. * step_size)),
    #                                disc_pos, grad)

    p_curr = jax.nn.sigmoid(theta)
    ind = jnp.array(u < p_curr)
    pos_new = (1. - disc_pos) * ind + disc_pos * (1. - ind)

    return pos_new, pstate

def take_contin_step(rng_key: PRNGKey, state: SGLDState, contin_grad_fn: Callable,
                     batch: Batch, step_size_fn: Callable) -> Tuple[PyTree, PreconditionState]:
    key_integrator, key_rmh = jax.random.split(rng_key)

    """Preconditioned SGLD update for the continuous variable. Ref: Li et.al 2015 (https://arxiv.org/abs/1512.07666) - 
        - Algorithm 1"""

    contin_pos = state.position
    precond = state.precond
    step_size = step_size_fn(state.count)
    noise = generate_gaussian_noise(key_integrator, contin_pos)
    grad, pstate = contin_grad_fn(contin_pos, precond, batch)

    m = jax.tree_util.tree_map(lambda k: 1./(EPS + jnp.sqrt(k)), pstate.v)
    noise = jax.tree_util.tree_map(lambda n, k: n*k*step_size, noise, m)
    grad = jax.tree_util.tree_map(lambda g, k: g*k, grad, m)
    new_pos = jax.tree_util.tree_map(
        lambda p, g, n: p + step_size * g + jnp.sqrt(2 * step_size) * n,
        contin_pos,
        grad,
        noise,
    )


    return new_pos, pstate

def take_mixed_contin_step(rng_key: PRNGKey, state: MixedState, contin_grad_fn: Callable,
                           batch: Batch, step_size_fn: Callable) -> Tuple[PyTree, PreconditionState]:

    """The same as above but the log-probability depends on a discrete r.v as we're working with mixed distribution"""

    key_integrator, key_rmh = jax.random.split(rng_key)

    disc_pos, contin_pos = state.discrete_position, state.contin_position
    precond = state.contin_precond
    step_size = step_size_fn(state.count)
    noise = generate_gaussian_noise(key_integrator, contin_pos)
    grad, pstate = contin_grad_fn(contin_pos, disc_pos, precond, batch)

    m = jax.tree_util.tree_map(lambda k: 1./(EPS + jnp.sqrt(k)), pstate.v)
    noise = jax.tree_util.tree_map(lambda n, k: n*k*step_size, noise, m)
    grad = jax.tree_util.tree_map(lambda g, k: g*k, grad, m)
    new_pos = jax.tree_util.tree_map(
        lambda p, g, n: p + step_size * g + jnp.sqrt(2 * step_size) * n,
        contin_pos,
        grad,
        noise,
    )

    # new_pos = {"mu": jax.nn.softplus(pos["mu"]), "sigma": jax.nn.softplus(pos["sigma"]),
    #            "nn_params": pos["nn_params"]}

    return new_pos, pstate


def get_mixed_sgld_kernel(discrete_grad_est: Callable, contin_grad_est:Callable,
                          disc_step_size_fn: Callable, contin_step_size_fn: Callable):
    """
    Constructs the kernel for sampling from the mixed distribution
    :param discrete_grad_est: Gradient estimator for the discrete variable
    :param contin_grad_est: Gradient estimator for the contin variable
    :param disc_step_size_fn: Learning rate schedule for discrete variable
    :param contin_step_size_fn: Learning rate schedule for contin variable
    :return: The Gibbs sampler kernel
    """
    def one_step(rng_key: PRNGKey, state: MixedState, batch: Batch) -> MixedState:
        """
        Main function (kernel) that does the gibbs sampling
        :param rng_key: PRNGKey for controlled randomness
        :param state: Current state
        :param batch: Mini-Batch to calculate the gradient on
        :return: new state
        """
        # Evolve each variable in tandem and combine the results
        count = state.count
        # Take a step for the discrete variable - sample from p(discrete | contin, data)
        new_disc_pos, disc_pcond = take_discrete_step(rng_key, state, discrete_grad_est, batch, disc_step_size_fn)
        state = MixedState(count, new_disc_pos, state.contin_position, disc_pcond, state.contin_precond)
        # Take a step for the contin variable - sample from p(contin | new_discrete, data)
        new_contin_pos, contin_precond = take_mixed_contin_step(rng_key, state, contin_grad_est, batch, contin_step_size_fn)

        new_state = MixedState(count + 1, new_disc_pos, new_contin_pos, disc_pcond, contin_precond)

        return new_state

    return one_step

def get_sgld_kernel(contin_grad_est:Callable, contin_step_size_fn: Callable):

    def one_step(rng_key: PRNGKey, state: SGLDState, batch: Batch) -> SGLDState:
        # Take a step for the contin variable - sample from p(contin | data)
        count = state.count
        new_contin_pos, contin_precond = take_contin_step(rng_key, state, contin_grad_est, batch, contin_step_size_fn)

        new_state = SGLDState(count + 1, new_contin_pos, contin_precond)

        return new_state

    return one_step

def init_mixed_sgld(disc_position: PyTree, contin_position: PyTree,
                    alpha: float) -> MixedState:

    """Initialises a new mixed state"""
    disc_v = jax.tree_util.tree_map(jnp.zeros_like, disc_position)
    contin_v = jax.tree_util.tree_map(jnp.zeros_like, contin_position)
    disc_precond = PreconditionState(disc_v, alpha)
    contin_precond = PreconditionState(contin_v, alpha)

    return MixedState(0, disc_position, contin_position, disc_precond, contin_precond)

def init_sgld(contin_position: PyTree,
              alpha: float) -> SGLDState:
    """Initialises a new continuous r.v state"""
    contin_v = jax.tree_util.tree_map(jnp.zeros_like, contin_position)
    contin_precond = PreconditionState(contin_v, alpha)

    return SGLDState(0, contin_position, contin_precond)


def inference_loop_multiple_chains(rng_key, kernel, initial_state, train_data, batch_size, num_samples,
                                   num_cycles, beta=0.5, n_chains=1):
    data_size = train_data.x.shape[0]

    # @jax.jit
    def inner_step(state, key):
        idxs = jax.random.choice(key, jnp.arange(data_size), (batch_size,), replace=False)
        batches = make_batch(idxs, train_data.x, train_data.y)
        state = kernel(key, state, batches)
        return state, state

    # @jax.jit
    # def cyclical_step(state, key):
    #     n_samples = (num_cycles - int(num_cycles * beta)) # number of sampling steps in single cycle
    #     n_exp = num_cycles - n_samples # number of exploration steps in single cycle
    #     exp_keys = jax.random.split(key, n_exp)
    #     sample_keys = jax.random.split(key, n_samples)
    #
    #     state, _ = jax.lax.scan(inner_step, state, exp_keys) #exploration stage
    #     state, states = jax.lax.scan(inner_step, state, sample_keys) #sampling stage
    #     return state, states



    _, key_samples = jax.random.split(rng_key, 2)
    # num_cycles = (num_samples + 1) // num_cycles
    # keys = jax.random.split(key_samples, num_cycles)
    # _, states = jax.lax.scan(cyclical_step, initial_state, keys)

    assert len(initial_state) == n_chains

    chain_disc_states = []
    chain_contin_states = []

    chain_keys = jax.random.split(key_samples, n_chains)
    for c in range(n_chains):
        keys = jax.random.split(chain_keys[c], num_samples)
        disc_states = []
        contin_states = []
        state = initial_state[c]
        M = num_samples // num_cycles
        for i, key in enumerate(keys):
            t = (i % M) / M
            state, _ = inner_step(state, key)
            if t > beta:
                disc_states.append(state.discrete_position)
                contin_states.append(state.contin_position)

        chain_disc_states.append(disc_states)
        chain_contin_states.append(contin_states)

    return chain_disc_states, chain_contin_states

def make_init_mixed_state(key, model, alpha, dim, data, batch_size):

    data_size = data.x.shape[0]
    disc_pos = tfd.Bernoulli(probs=0.5).sample(seed=key, sample_shape=(dim,))*1.

    init_idx = jax.random.choice(key, jnp.arange(data_size), shape=(batch_size, ), replace=False)
    init_batch = Batch(data.x[init_idx], data.y[init_idx])
    params = model.init(key, init_batch.x, disc_pos)

    # sigma = tfd.InverseGamma(sigma_a, 1.).sample(seed=key)
    # mu = tfd.InverseGamma(mu_a, 1.).sample(seed=key)

    # params = {"mu": mu, "sigma": sigma, "nn_params": nn_params}

    return init_mixed_sgld(disc_pos, params, alpha)

def make_init_sgld_state(key, model, alpha, data, batch_size):
    data_size = data.x.shape[0]

    init_idx = jax.random.choice(key, jnp.arange(data_size), shape=(batch_size, ), replace=False)
    init_batch = Batch(data.x[init_idx], data.y[init_idx])
    params = model.init(key, init_batch.x)

    return init_sgld(params, alpha)

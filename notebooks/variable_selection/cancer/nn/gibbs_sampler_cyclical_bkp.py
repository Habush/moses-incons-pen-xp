## Author Abdulrahman S. Omar <hsamireh@gmail>

from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from blackjax.mcmc.diffusions import generate_gaussian_noise
from blackjax.types import PyTree, PRNGKey
import tree_utils
from nn_util import *
from tqdm import tqdm

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

    disc_logprob: PyTree
    contin_logprob: PyTree

    disc_logprob_grad: PyTree
    contin_logprob_grad: PyTree


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


def generate_disc_logprior_fn(J, eta, mu, temperature):

    def discrete_logprior_fn(gamma):
        log_prior = gamma_energy(gamma, J, eta, mu)
        return log_prior / temperature

    return discrete_logprior_fn

def generate_contin_logprior_fn(sigma, temperature):

    def contin_logprior_fn(params):
        """Computes the Gaussian prior log-density."""

        weight_decay = (1./sigma)*temperature
        n_params = sum([p.size for p in jax.tree_util.tree_leaves(params)])
        log_prob = -(0.5 * tree_utils.tree_dot(params, params) * weight_decay +
                     0.5 * n_params * jnp.log(weight_decay / (2 * jnp.pi)))

        return log_prob

    return contin_logprior_fn

def make_gaussian_likelihood(sigma, temperature, data_size):
    weight_decay = (1./sigma)*temperature

    def gaussian_log_likelihood(model, params, batch, gamma):
        x, y = batch
        preds = model.apply(params, x, gamma).ravel()
        se = (preds - y)**2
        log_likelihood = -(0.5 * se * weight_decay +
                            0.5 * jnp.log(weight_decay / ( 2 * jnp.pi)))

        return data_size*jnp.mean(log_likelihood)

    return gaussian_log_likelihood

def make_clf_log_likelihood(temperature, data_size):

    def clf_log_likelihood(model, params, batch, gamma):
        x, y = batch
        logits = model.apply(params, x, gamma)
        num_classes = logits.shape[-1]
        labels = jax.nn.one_hot(y, num_classes)
        log_likelihood = (labels * jax.nn.log_softmax(logits)) / temperature

        return data_size*jnp.mean(log_likelihood)
    
    return clf_log_likelihood


def update_grad(grad_prior, grad_ll, state, data_size, temp):
    v = jax.tree_util.tree_map(lambda v, g: state.alpha*v + (1 - state.alpha)*(g**2),
                               state.v, grad_ll) # Calcuate the Exponentially Moving Weight Average (EWMA)
    grad = jax.tree_util.tree_map(lambda gp, gl: temp*((gp/data_size) + gl), # Acc. Equation 7 of Wenzel et.al 2020 re-scale the prior by 1/n, n - training size
                                  grad_prior, grad_ll)

    return grad, PreconditionState(v, state.alpha)

def generate_discrete_grad_estimator(model, logprior_fn, log_likelihood_fn):

    def grad_estimator(params, gamma, batch):
        log_ll, grad_ll = jax.value_and_grad(log_likelihood_fn, argnums=3)(model, params, batch, gamma)
        log_prior, grad_prior = jax.value_and_grad(logprior_fn)(gamma)
        log_prob = jax.tree_util.tree_map(lambda lp, ll: lp + ll, log_prior, log_ll)
        log_prob_grad = jax.tree_util.tree_map(lambda gp, gl: gp + gl, 
                                  grad_prior, grad_ll)
        return log_prob, log_prob_grad

    return grad_estimator



def generate_mixed_contin_grad_estimator(model, logprior_fn, log_likelihood_fn):

    def grad_estimator(params, gamma, batch):
        log_ll, grad_ll = jax.value_and_grad(log_likelihood_fn, argnums=1)(model, params, batch, gamma)
        log_prior, grad_prior = jax.value_and_grad(logprior_fn)(params)
        log_prob = jax.tree_util.tree_map(lambda lp, ll: lp + ll, log_prior, log_ll)
        log_prob_grad = jax.tree_util.tree_map(lambda gp, gl: gp + gl, 
                                  grad_prior, grad_ll)
        return log_prob, log_prob_grad

    return grad_estimator


def proposal_noisy(key, theta, grad_theta, step_size):
  diff = (grad_theta*-(2*theta - 1)) - (1./(2*step_size))
  delta = jax.random.bernoulli(key, jax.nn.sigmoid(diff))
  theta_delta = (1 - theta)*delta + theta*(1 - delta)
  return theta_delta*1.

def proposal(theta, grad_theta, step_size):
  diff = (grad_theta*-(2*theta - 1)) - (1./(2*step_size))
  prob = jax.nn.sigmoid(diff)
  prob_inv = 1 - prob
  prob = prob[...,None]
  prob_inv = prob_inv[...,None]
  delta = jnp.argmax(jnp.concatenate([prob, prob_inv], axis=1), axis=-1)  

  theta_delta = (1 - theta)*delta + theta*(1 - delta)
  return theta_delta*1.


def take_discrete_step(rng_key: PRNGKey, state: MixedState, disc_grad_fn: Callable,
                       batch: Tuple[np.ndarray, np.ndarray], step_size: float) -> Tuple[PyTree, PyTree, PyTree]:

    """SGLD update for the discrete variable. Ref: Zhang et.al 2022 (https://arxiv.org/abs/2206.09914) - Alogrithm 2"""
    disc_pos, contin_pos = state.discrete_position, state.contin_position
    logprob, logrpob_grad = disc_grad_fn(contin_pos, disc_pos, batch)
    new_pos = proposal(state.discrete_position, logrpob_grad, step_size)

    return new_pos, logprob, logrpob_grad

def take_discrete_step_noisy(rng_key: PRNGKey, state: MixedState, disc_grad_fn: Callable,
                            batch: Tuple[np.ndarray, np.ndarray], step_size: float) -> Tuple[PyTree, PyTree, PyTree]:

    """SGLD update for the discrete variable. Ref: Zhang et.al 2022 (https://arxiv.org/abs/2206.09914) - Alogrithm 2"""

    _, key_rmh, key_accept = jax.random.split(rng_key, 3)

    disc_pos, contin_pos = state.discrete_position, state.contin_position
    logprob, logrpob_grad = disc_grad_fn(contin_pos, disc_pos, batch)
    new_pos = proposal_noisy(key_rmh, state.discrete_position, logrpob_grad, step_size)

    return new_pos, logprob, logrpob_grad

def take_mixed_contin_step(rng_key: PRNGKey, state: MixedState, contin_grad_fn: Callable,
                           batch: Tuple[np.ndarray, np.ndarray], 
                           step_size: float) -> Tuple[PyTree, PyTree, PyTree]:

    """The same as above but the log-probability depends on a discrete r.v as we're working with mixed distribution"""

    key_integrator, key_rmh = jax.random.split(rng_key)

    disc_pos, contin_pos = state.discrete_position, state.contin_position
    logprob, logprob_grad = contin_grad_fn(contin_pos, disc_pos, batch)
    params = jax.tree_util.tree_map(lambda p, g: p + step_size*g, 
                                            contin_pos, logprob_grad)

    return params, logprob, logprob_grad

def take_mixed_contin_step_noisy(rng_key: PRNGKey, state: MixedState, contin_grad_fn: Callable,
                           batch: Tuple[np.ndarray, np.ndarray],
                           step_size: float) -> Tuple[PyTree, PyTree, PyTree]:

    """The same as above but the log-probability depends on a discrete r.v as we're working with mixed distribution"""

    key_integrator, key_rmh = jax.random.split(rng_key)

    disc_pos, contin_pos = state.discrete_position, state.contin_position
    logprob, logprob_grad = contin_grad_fn(contin_pos, disc_pos, batch)
    noise = generate_gaussian_noise(key_integrator, contin_pos)
    params = jax.tree_util.tree_map(lambda p, g, n: p + step_size*g + (jnp.sqrt(2 * step_size)*n), 
                                            contin_pos, logprob_grad, noise)

    return params, logprob, logprob_grad


def get_mixed_sgld_kernel(discrete_grad_est: Callable, contin_grad_est:Callable) -> Callable:
    """
    Constructs the kernel for sampling from the mixed distribution
    :param discrete_grad_est: Gradient estimator for the discrete variable
    :param contin_grad_est: Gradient estimator for the contin variable
    :param disc_step_size_fn: Learning rate schedule for discrete variable
    :param contin_step_size_fn: Learning rate schedule for contin variable
    :return: The Gibbs sampler kernel
    """
    def one_step(rng_key: PRNGKey, state: MixedState, batch: Tuple[np.ndarray, np.ndarray], 
                    disc_step_size: float, contin_step_size: float) -> MixedState:
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
        new_disc_pos, disc_logprob, disc_logprob_grad = take_discrete_step(rng_key, state, discrete_grad_est, batch, disc_step_size)

        state = MixedState(count, new_disc_pos, state.contin_position, 
                                    disc_logprob, state.contin_logprob, disc_logprob_grad, state.contin_logprob_grad)

        # Take a step for the contin variable - sample from p(contin | new_discrete, data)
        new_contin_pos, contin_logprob, contin_logprob_grad = take_mixed_contin_step(rng_key, state, contin_grad_est, batch, contin_step_size)

        new_state = MixedState(count + 1, new_disc_pos, new_contin_pos, disc_logprob, contin_logprob, 
                                                    disc_logprob_grad, contin_logprob_grad)

        return new_state

    return one_step

def get_mixed_sgld_kernel_noisy(discrete_grad_est: Callable, contin_grad_est:Callable) -> Callable:
    """
    Constructs the kernel for sampling from the mixed distribution
    :param discrete_grad_est: Gradient estimator for the discrete variable
    :param contin_grad_est: Gradient estimator for the contin variable
    :param disc_step_size_fn: Learning rate schedule for discrete variable
    :param contin_step_size_fn: Learning rate schedule for contin variable
    :return: The Gibbs sampler kernel
    """
    def one_step(rng_key: PRNGKey, state: MixedState, batch: Tuple[np.ndarray, np.ndarray],
                    disc_step_size: float, contin_step_size: float, temp: float) -> MixedState:
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
        new_disc_pos, disc_logprob, disc_logprob_grad = take_discrete_step_noisy(rng_key, state, discrete_grad_est, batch, disc_step_size)

        state = MixedState(count, new_disc_pos, state.contin_position, 
                                    disc_logprob, state.contin_logprob, disc_logprob_grad, state.contin_logprob_grad)

        # Take a step for the contin variable - sample from p(contin | new_discrete, data)
        new_contin_pos, contin_logprob, contin_logprob_grad = take_mixed_contin_step_noisy(rng_key, state, contin_grad_est, batch, contin_step_size)

        new_state = MixedState(count + 1, new_disc_pos, new_contin_pos, disc_logprob, contin_logprob, 
                                                    disc_logprob_grad, contin_logprob_grad)

        return new_state

    return one_step


def make_init_mixed_state(key, model, disc_grad_est_fn, contin_grad_est_fn, data_loader, batch_size):

    state = None
    for batch in data_loader:
        x, y = batch
        dim = x.shape[-1]
        disc_pos = tfd.Bernoulli(probs=0.5).sample(seed=key, sample_shape=(dim,))*1.
        params = model.init(key, x, disc_pos)
        disc_logprob, disc_logprob_grad = disc_grad_est_fn(params, disc_pos, batch)
        contin_logprob, contin_logprob_grad = contin_grad_est_fn(params, disc_pos, batch)

        state = MixedState(0, disc_pos, params, disc_logprob, contin_logprob, 
                                                disc_logprob_grad, contin_logprob_grad)
        break

    return state

def adjust_learning_rate(lr_0, epoch, total, num_cycles, num_batch, batch_idx):
    rcounter = epoch*num_batch+batch_idx
    k = total // num_cycles
    rk = (rcounter % k) / k
    cos_inner = np.pi * rk
    cos_out = np.cos(cos_inner) + 1
    lr = 0.5*cos_out*lr_0

    return lr, rk

def inference_loop_multiple_chains(rng_key, kernel, kernel_noisy, init_state, disc_lr, contin_lr,
                                   data_loader, batch_size, num_samples,
                                   num_cycles, temp, beta=0.5):

    _, key_samples = jax.random.split(rng_key, 2)

    num_batch = len(data_loader)
    print(f"Num batches: {num_batch}")
    total = num_samples*num_batch

    keys = jax.random.split(rng_key, num_samples)
    states = []
    exp_states = []

    state = init_state
    for epoch, key in enumerate(keys):
        for batch_idx, batch in enumerate(data_loader):
            cur_disc_lr, _ = adjust_learning_rate(disc_lr, epoch, total, num_cycles, num_batch, batch_idx)
            cur_contin_lr, rk = adjust_learning_rate(contin_lr, epoch, total, num_cycles, num_batch, batch_idx)

            if rk < beta:
                state = kernel_noisy(key, state, batch, cur_disc_lr, cur_contin_lr)
                exp_states.append(state)
            else:
                state = kernel_noisy(key, state, batch, cur_disc_lr, cur_contin_lr, temp)
                states.append(state)

    return states, exp_states

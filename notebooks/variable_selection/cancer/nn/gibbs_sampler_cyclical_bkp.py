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

class OptaxState(NamedTuple):

    disc_state: optax.OptState
    contin_state: optax.OptState

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

        # return data_size*jnp.mean(log_likelihood)
        return log_likelihood

    return gaussian_log_likelihood

def make_clf_log_likelihood(temperature, data_size):

    def clf_log_likelihood(model, params, batch, gamma):
        x, y = batch
        logits = model.apply(params, x, gamma)
        num_classes = logits.shape[-1]
        labels = jax.nn.one_hot(y, num_classes)
        log_likelihood = (labels * jax.nn.log_softmax(logits)) / temperature

        # return data_size*jnp.mean(log_likelihood)
        return log_likelihood
    
    return clf_log_likelihood


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
  diff = (0.5*grad_theta*-(2*theta - 1)) - (1./(2*step_size))
  delta = jax.random.bernoulli(key, jax.nn.sigmoid(diff))
  theta_delta = (1 - theta)*delta + theta*(1 - delta)
  return theta_delta*1.

def proposal(theta, grad_theta, step_size):
  diff = (0.5*grad_theta*-(2*theta - 1)) - (1./(2*step_size))
  prob = jax.nn.sigmoid(diff)
  prob_inv = 1 - prob
  prob = prob[...,None]
  prob_inv = prob_inv[...,None]
  delta = jnp.argmax(jnp.concatenate([prob, prob_inv], axis=1), axis=-1)  

  theta_delta = (1 - theta)*delta + theta*(1 - delta)
  return theta_delta*1.


def take_discrete_step(rng_key: PRNGKey, state: MixedState, opt_state: optax.OptState,
                       disc_grad_fn: Callable, disc_optim: optax.GradientTransformation,
                       batch: Tuple[np.ndarray, np.ndarray], step_size: float) -> Tuple[PyTree, PyTree, PyTree, optax.OptState]:

    """SGLD update for the discrete variable. Ref: Zhang et.al 2022 (https://arxiv.org/abs/2206.09914) - Alogrithm 2"""
    disc_pos, contin_pos = state.discrete_position, state.contin_position
    p = disc_pos.shape[-1]
    logprob, logprob_grad = jnp.zeros(p), jnp.zeros(p)
    # logprob, logprob_grad = disc_grad_fn(contin_pos, disc_pos, batch)
    # updates, opt_state = disc_optim.update(logprob_grad, opt_state)
    # new_pos = proposal(state.discrete_position, updates, step_size)
    new_pos = jnp.ones(p)
    return new_pos, logprob, logprob_grad, opt_state

def take_discrete_step_noisy(rng_key: PRNGKey, state: MixedState, opt_state: optax.OptState,
                            disc_grad_fn: Callable, disc_optim: optax.GradientTransformation,
                            batch: Tuple[np.ndarray, np.ndarray], step_size: float) -> Tuple[PyTree, PyTree, PyTree, optax.OptState]:

    """SGLD update for the discrete variable. Ref: Zhang et.al 2022 (https://arxiv.org/abs/2206.09914) - Alogrithm 2"""

    # _, key_rmh, key_accept = jax.random.split(rng_key, 3)

    disc_pos, contin_pos = state.discrete_position, state.contin_position
    # logprob, logprob_grad = disc_grad_fn(contin_pos, disc_pos, batch)
    # updates, opt_state = disc_optim.update(logprob_grad, opt_state)
    # new_pos = proposal_noisy(key_rmh, state.discrete_position, updates, step_size)

    # return new_pos, logprob, logprob_grad, opt_state
    disc_pos, contin_pos = state.discrete_position, state.contin_position
    p = disc_pos.shape[-1]
    logprob, logprob_grad = jnp.zeros(p), jnp.zeros(p)
    new_pos = jnp.ones(p)
    return new_pos, logprob, logprob_grad, opt_state


def take_mixed_contin_step(rng_key: PRNGKey, state: MixedState, opt_state: optax.OptState,
                           contin_grad_fn: Callable, contin_optim: optax.GradientTransformation,
                           batch: Tuple[np.ndarray, np.ndarray], 
                           step_size: float) -> Tuple[PyTree, PyTree, PyTree, optax.OptState]:

    """The same as above but the log-probability depends on a discrete r.v as we're working with mixed distribution"""

    key_integrator, key_rmh = jax.random.split(rng_key)

    disc_pos, contin_pos = state.discrete_position, state.contin_position
    logprob, logprob_grad = contin_grad_fn(contin_pos, disc_pos, batch)
    updates, opt_state = contin_optim.update(logprob_grad, opt_state)
    
    params = optax.apply_updates(contin_pos, updates)

    return params, logprob, logprob_grad, opt_state

def take_mixed_contin_step_noisy(rng_key: PRNGKey, state: MixedState, opt_state: optax.OptState,
                                contin_grad_fn: Callable, contin_optim: optax.GradientTransformation,
                                batch: Tuple[np.ndarray, np.ndarray], 
                                step_size: float) -> Tuple[PyTree, PyTree, PyTree, optax.OptState]:

    """The same as above but the log-probability depends on a discrete r.v as we're working with mixed distribution"""

    key_integrator, key_rmh = jax.random.split(rng_key)

    disc_pos, contin_pos = state.discrete_position, state.contin_position
    logprob, logprob_grad = contin_grad_fn(contin_pos, disc_pos, batch)
    updates, opt_state = contin_optim.update(logprob_grad, opt_state)
    params = optax.apply_updates(contin_pos, updates)

    noise = generate_gaussian_noise(key_integrator, contin_pos)
    params = jax.tree_util.tree_map(lambda p, n: p + (jnp.sqrt(2 * step_size)*n), 
                                            params, noise) # Add noise

    return params, logprob, logprob_grad, opt_state



def get_mixed_sgld_kernel(discrete_grad_est: Callable, contin_grad_est:Callable, 
                                disc_optim: optax.GradientTransformation, contin_optim: optax.GradientTransformation) -> Callable:
    """
    Constructs the kernel for sampling from the mixed distribution
    :param discrete_grad_est: Gradient estimator for the discrete variable
    :param contin_grad_est: Gradient estimator for the contin variable
    :param disc_step_size_fn: Learning rate schedule for discrete variable
    :param contin_step_size_fn: Learning rate schedule for contin variable
    :return: The Gibbs sampler kernel
    """
    def one_step(rng_key: PRNGKey, state: MixedState, opt_state: OptaxState, batch: Tuple[np.ndarray, np.ndarray],
                    disc_step_size: float, contin_step_size: float) -> Tuple[MixedState, OptaxState]:
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
        new_disc_pos, disc_logprob, disc_logprob_grad, disc_opt_state = take_discrete_step(rng_key, state, opt_state.disc_state,
                                                                                  discrete_grad_est, disc_optim, 
                                                                                  batch, disc_step_size)

        state = MixedState(count, new_disc_pos, state.contin_position, 
                                    disc_logprob, state.contin_logprob, disc_logprob_grad, state.contin_logprob_grad)

        # Take a step for the contin variable - sample from p(contin | new_discrete, data)
        new_contin_pos, contin_logprob, contin_logprob_grad, contin_opt_state = take_mixed_contin_step(rng_key, state, opt_state.contin_state, 
                                                                                    contin_grad_est, contin_optim,
                                                                                    batch, contin_step_size)

        new_state = MixedState(count + 1, new_disc_pos, new_contin_pos, disc_logprob, contin_logprob, 
                                                    disc_logprob_grad, contin_logprob_grad)

        opt_state = OptaxState(disc_opt_state, contin_opt_state)

        return new_state, opt_state

    return one_step

def get_mixed_sgld_kernel_noisy(discrete_grad_est: Callable, contin_grad_est:Callable, 
                                disc_optim: optax.GradientTransformation, contin_optim: optax.GradientTransformation) -> Callable:
    """
    Constructs the kernel for sampling from the mixed distribution
    :param discrete_grad_est: Gradient estimator for the discrete variable
    :param contin_grad_est: Gradient estimator for the contin variable
    :param disc_step_size_fn: Learning rate schedule for discrete variable
    :param contin_step_size_fn: Learning rate schedule for contin variable
    :return: The Gibbs sampler kernel
    """
    def one_step(rng_key: PRNGKey, state: MixedState, opt_state: OptaxState, batch: Tuple[np.ndarray, np.ndarray],
                    disc_step_size: float, contin_step_size: float) -> Tuple[MixedState, OptaxState]:
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
        new_disc_pos, disc_logprob, disc_logprob_grad, disc_opt_state = take_discrete_step_noisy(rng_key, state, opt_state.disc_state,
                                                                                  discrete_grad_est, disc_optim, 
                                                                                  batch, disc_step_size)

        state = MixedState(count, new_disc_pos, state.contin_position, 
                                    disc_logprob, state.contin_logprob, disc_logprob_grad, state.contin_logprob_grad)

        # Take a step for the contin variable - sample from p(contin | new_discrete, data)
        new_contin_pos, contin_logprob, contin_logprob_grad, contin_opt_state = take_mixed_contin_step_noisy(rng_key, state, opt_state.contin_state, 
                                                                                    contin_grad_est, contin_optim,
                                                                                    batch, contin_step_size)

        new_state = MixedState(count + 1, new_disc_pos, new_contin_pos, disc_logprob, contin_logprob, 
                                                    disc_logprob_grad, contin_logprob_grad)
        opt_state = OptaxState(disc_opt_state, contin_opt_state)

        return new_state, opt_state

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

def make_cyclical_lr_fn(lr_0, total, num_cycles):
    k = total // num_cycles
    def schedule_fn(step):
        rk = (step % k) / k
        cos_inner = jnp.pi * rk
        cos_out = jnp.cos(cos_inner) + 1
        lr = 0.5*cos_out*lr_0

        return lr

    return schedule_fn

def inference_loop_multiple_chains(rng_key, model, disc_lr, contin_lr,
                                   J, temp, sigma, eta, mu,
                                   data_loader, batch_size, num_samples,
                                   num_cycles, beta=0.5, classifier=True):

    key_init, key_samples = jax.random.split(rng_key, 2)

    num_batch = len(data_loader)
    total = num_samples*num_batch
    data_size = len(data_loader.dataset)

    # Setup prior and likelihood functions

    if classifier:
        log_likelihood_fn = make_clf_log_likelihood(temp, data_size)
    else:
        log_likelihood_fn = make_gaussian_likelihood(sigma, temp, data_size)

    disc_logprior_fn = generate_disc_logprior_fn(J, mu, eta, temp)
    contin_logprior_fn = generate_contin_logprior_fn(sigma, temp)
    disc_grad_est_fn = generate_discrete_grad_estimator(model, disc_logprior_fn, log_likelihood_fn)
    contin_grad_est_fn = generate_mixed_contin_grad_estimator(model, contin_logprior_fn, log_likelihood_fn)

    init_state = make_init_mixed_state(key_init, model, disc_grad_est_fn, contin_grad_est_fn, 
                                                data_loader, batch_size)

    # Setup optimizer and scheduling functions
    cycle_len = total // num_cycles

    disc_schedule_fn = make_cyclical_lr_fn(disc_lr, total, num_cycles)
    contin_schedule_fn = make_cyclical_lr_fn(contin_lr, total, num_cycles)

    disc_optim = optax.chain(
                    optax.clip_by_global_norm(5.0),
                    optax.scale_by_adam(),
                    optax.scale_by_schedule(disc_schedule_fn))

    contin_optim = optax.chain(
                    optax.clip_by_global_norm(5.0),
                    optax.scale_by_adam(),
                    optax.scale_by_schedule(contin_schedule_fn))

    disc_opt_state, contin_opt_state = disc_optim.init(init_state.discrete_position),  \
                                       contin_optim.init(init_state.contin_position)

    
    opt_state = OptaxState(disc_opt_state, contin_opt_state)
    state = init_state
    # Setup kernels

    kernel = jax.jit(get_mixed_sgld_kernel(disc_grad_est_fn, contin_grad_est_fn, 
                                            disc_optim, contin_optim))

    kernel_noisy = jax.jit(get_mixed_sgld_kernel_noisy(disc_grad_est_fn, contin_grad_est_fn,
                                                disc_optim, contin_optim))


            

    keys = jax.random.split(rng_key, num_samples)
    states = []
    exp_states = []
    step = 0
    for epoch, key in enumerate(keys):
        for batch_idx, batch in enumerate(data_loader):
            rk = (step % cycle_len) / cycle_len
            cur_disc_lr = disc_schedule_fn(step)
            cur_contin_lr  = contin_schedule_fn(step)

            if rk < beta:
                if num_batch == 1:
                    state, opt_state = kernel_noisy(key, state, opt_state, batch, 
                                                        cur_disc_lr, cur_contin_lr)
                else:
                    state, opt_state = kernel(key, state, opt_state, batch, 
                                                cur_disc_lr, cur_contin_lr)
                    
                exp_states.append(state)
            else:
                state, opt_state = kernel_noisy(key, state, opt_state, batch, 
                                            cur_disc_lr, cur_contin_lr)
                states.append(state)
            
            step += 1

    return states, exp_states

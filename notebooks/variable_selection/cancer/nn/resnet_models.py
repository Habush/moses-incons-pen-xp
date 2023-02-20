import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
import haiku as hk
import optax
tfd = tfp.distributions
from optim_util import *

class ResnetState(NamedTuple):
    params: hk.Params
    opt_state: hk.Params
    net_state: hk.State

class BgResnetState(NamedTuple):
    params: hk.Params
    gamma: jnp.DeviceArray
    opt_state: hk.Params
    disc_opt_state: hk.Params
    net_state: hk.State

class ResNetBlock(hk.Module):
    def __init__(self, act_fn, dim, init_fn, dropout_rate, name=None):
        super().__init__(name)
        self.act_fn = act_fn
        self.dim = dim
        self.init_fn = init_fn
        self.dropout_rate = dropout_rate

    def __call__(self, x, is_training):
        key = hk.next_rng_key()
        subkey1, subkey2 = jax.random.split(key, 2)
        z = hk.Linear(self.dim, w_init=self.init_fn, with_bias=False)(x)
        z = hk.BatchNorm(True, True, 0.9)(z, is_training)
        z = self.act_fn(z)
        if is_training:
            z = hk.dropout(subkey1, self.dropout_rate, z)
        z = hk.Linear(self.dim, w_init=self.init_fn, with_bias=False)(z)
        if is_training:
            z = hk.dropout(subkey2, self.dropout_rate, z)
        x_out = self.act_fn(z + x)
        return x_out

class PreActResNetBlock(ResNetBlock):

    def __init__(self, act_fn, dim, init_fn, dropout_rate, name=None):
        super().__init__(act_fn, dim, init_fn, dropout_rate, name)


    def __call__(self, x, is_training):
        key = hk.next_rng_key()
        subkey1, subkey2, subkey3 = jax.random.split(key, 3)
        z = hk.BatchNorm(True, True, 0.9)(x, is_training)
        z = self.act_fn(z)
        if is_training:
            z = hk.dropout(subkey1, self.dropout_rate, z)
        z = hk.Linear(self.dim, w_init=self.init_fn, with_bias=False)(z)
        z = hk.BatchNorm(True, True, 0.9)(z, is_training)
        z = self.act_fn(z)
        if is_training:
            z = hk.dropout(subkey2, self.dropout_rate, z)
        z = hk.Linear(self.dim, w_init=self.init_fn, with_bias=False)(z)
        if is_training:
            z = hk.dropout(subkey3, self.dropout_rate, z)
        x_out = z + x
        return x_out


class ResNet:
    def __init__(self, block_class, num_blocks, hidden_dims,
                 optim, init_fn, act_fn, dropout_rate):

        self.block_class = block_class
        self.num_blocks = num_blocks
        self.hidden_dims = hidden_dims
        self.act_fn = act_fn
        self.optimiser = optim
        self.init_fn = init_fn #TODO try d/t values
        self.dropout_rate = dropout_rate
        self._forward = hk.transform_with_state(self._forward_fn)
        self.loss = jax.jit(self.loss)
        self.update = jax.jit(self.update)

        assert len(self.num_blocks) == len(self.hidden_dims)

    def init(self, rng, x):
        params, net_state = self._forward.init(rng, x, is_training=True)
        opt_state = self.optimiser.init(params)
        return ResnetState(params, opt_state, net_state)

    def apply(self, state, x, key, is_training=True):
        return self._forward.apply(state.params, state.net_state, key, x, is_training)


    def update(self, key, train_state, x, y):
        params, opt_state, net_state = train_state
        grads, net_state = jax.grad(self.loss, has_aux=True)(params, net_state, key, x, y)
        updates, opt_state = self.optimiser.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return ResnetState(params, opt_state, net_state)

    def _forward_fn(self, x, is_training):

        # First layer
        x = hk.Linear(self.hidden_dims[0], w_init=self.init_fn, with_bias=False)(x)

        if self.block_class == ResNetBlock: # If pre-activation block , we don't apply non-linearities  yet
            x = hk.BatchNorm(True, True, 0.9)(x, is_training)
            x = self.act_fn(x)

        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                x = self.block_class(self.act_fn, self.hidden_dims[block_idx], self.init_fn, self.dropout_rate)(x, is_training)


        x = hk.Linear(1)(x)

        return x

    def loss(self, params, net_state, key, x, y):
        preds, state = self._forward.apply(params, net_state, key, x, True)
        preds = preds.squeeze()
        nll_loss = jnp.mean((preds - y)**2)

        return nll_loss, state


class BgResNet:
    def __init__(self, block_class, num_blocks, hidden_dims,
                 optim, disc_optim, init_fn, act_fn, dropout_rate,
                 eta, mu, J, data_size):

        self.block_class = block_class
        self.num_blocks = num_blocks
        self.hidden_dims = hidden_dims
        self.act_fn = act_fn
        self.optimiser = optim
        self.disc_optimiser = disc_optim
        self.init_fn = init_fn #TODO try d/t values
        self.dropout_rate = dropout_rate
        self._forward = hk.transform_with_state(self._forward_fn)
        self.loss = jax.jit(self.loss)
        self.disc_loss = jax.jit(self.disc_loss)
        self.update = jax.jit(self.update)

        self.eta = eta
        self.mu = mu
        self.J = J
        self.data_size = data_size


    def init(self, rng, x):
        gamma = tfd.Bernoulli(0.5*jnp.ones(x.shape[-1])).sample(seed=rng)*1.
        params, net_state = self._forward.init(rng, x, gamma, is_training=True)
        opt_state = self.optimiser.init(params)
        disc_opt_state = self.disc_optimiser.init(gamma)
        return BgResnetState(params, gamma, opt_state, disc_opt_state, net_state)

    def apply(self, state, x, key, is_training=True):
        return self._forward.apply(state.params, state.net_state, key, x, state.gamma, is_training)


    def update(self, key, train_state, x, y):
        params, gamma, opt_state, disc_opt_state, net_state = train_state
        grads, net_state = jax.grad(self.loss, has_aux=True)(params, net_state, key, gamma, x, y)
        updates, opt_state = self.optimiser.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        disc_grads, net_state = jax.grad(self.disc_loss, has_aux=True)(gamma, params, net_state, key, x, y)
        gamma, disc_opt_state = self.disc_optimiser.update(key, gamma, disc_grads, disc_opt_state)

        return BgResnetState(params, gamma, opt_state, disc_opt_state, net_state)

    def _forward_fn(self, x, gamma, is_training):
        # if is_training:
        x = x @ jnp.diag(gamma)
        # First layer
        x = hk.Linear(self.hidden_dims[0], w_init=self.init_fn, with_bias=False)(x)

        if self.block_class == ResNetBlock: # If pre-activation block , we don't apply non-linearities  yet
            x = hk.BatchNorm(True, True, 0.9)(x, is_training)
            x = self.act_fn(x)

        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                x = self.block_class(self.act_fn, self.hidden_dims[block_idx], self.init_fn, self.dropout_rate)(x, is_training)


        x = hk.Linear(1)(x)

        return x

    def log_likelihood(self, params, net_state, key, gamma, x, y):
        preds, state = self._forward.apply(params, net_state, key, x, gamma, True)
        preds = preds.squeeze()
        nll_loss = jnp.mean((preds - y)**2)
        # log_prob = jnp.sum(tfd.Normal(preds, 1.0).log_prob(y))
        batch_size = x.shape[0]
        log_prob = (self.data_size/batch_size)*nll_loss

        return log_prob, state

    def loss(self, params, net_state, key, gamma, x, y):
        logprob_likelihood, net_state = self.log_likelihood(params, net_state, key, gamma, x, y)
        return logprob_likelihood, net_state

    def disc_loss(self, gamma, params, net_state, key, x, y):
        log_prior = self.ising_prior(gamma)
        log_ll, net_state = self.loss(params, net_state, key, gamma, x, y)
        return log_prior - log_ll, net_state

    def ising_prior(self, gamma):
        """Log probability of the Ising model - prior over the discrete variables"""
        return (0.5*self.eta*(gamma.T @ self.J @ gamma) - self.mu*jnp.sum(gamma))
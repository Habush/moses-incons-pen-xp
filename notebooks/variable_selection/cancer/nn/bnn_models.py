import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from optim_util import *

class MLP():
    def __init__(self, optim, hidden_sizes, init_fn, act_fn=jax.nn.relu):
        self.hidden_sizes = hidden_sizes
        self.act_fn = act_fn
        self._forward = hk.without_apply_rng(hk.transform(self._forward_fn))
        self.loss = jax.jit(self.loss)
        self.update = jax.jit(self.update)
        self.optimiser = optim
        self.init_fn = init_fn

    def init(self, rng, x):
        params = self._forward.init(rng, x)
        opt_state = self.optimiser.init(params)
        return params, opt_state

    def apply(self, params, x):
        return self._forward.apply(params, x).ravel()

    def update(self, params, opt_state, x, y):
        grads = jax.grad(self.loss)(params, x, y)
        updates, opt_state = self.optimiser.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    def _forward_fn(self, x):
        for hd in self.hidden_sizes:
            x = hk.Linear(hd, w_init=self.init_fn, b_init=jnp.zeros)(x)
            x = self.act_fn(x)

        x = hk.Linear(1)(x)
        return x

    def loss(self, params, x, y):
        preds = self.apply(params, x)
        return jnp.mean((preds - y)**2)


class BayesNN():
    def __init__(self, sgd_optim, sgld_optim, temperature, sigma, data_size, hidden_sizes, act_fn=jax.nn.relu):
        self.hidden_sizes = hidden_sizes
        self.act_fn = act_fn
        self.sgd_optim = sgd_optim
        self.sgld_optim = sgld_optim
        self.optimiser = sgd_optim
        self._forward = hk.without_apply_rng(hk.transform(self._forward_fn))
        self.loss = jax.jit(self.loss)
        self.update = jax.jit(self.update)

        self.temperature = temperature
        self.sigma = sigma
        self.data_size = data_size
        self.add_noise = False

        # weight_decay = self.sigma*self.temperature
        # self.weight_prior = tfd.Normal(0, self.sigma)
        self.weight_prior = tfd.StudentT(df=2, loc=0, scale=self.sigma)
        # self.weight_prior = tfd.Laplace(0, self.sigma)

    def init(self, rng, x):
        params = self._forward.init(rng, x)
        opt_state = self.optimiser.init(params)
        return params, opt_state

    def apply(self, params, x):
        return self._forward.apply(params, x).ravel()


    def update(self, key, params, opt_state, x, y):
        if self.add_noise:
            self.optimiser = self.sgld_optim
        else:
            self.optimiser = self.sgd_optim
        grads = jax.grad(self.loss)(params, x, y)
        updates, opt_state = self.optimiser.update(key, grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    def _forward_fn(self, x):
        init_fn = hk.initializers.VarianceScaling()
        for hd in self.hidden_sizes:
            x = hk.Linear(hd, w_init=init_fn, b_init=init_fn)(x)
            x = self.act_fn(x)

        x = hk.Linear(1)(x)
        return x

    def log_prior(self, params):
        """Computes the Gaussian prior log-density."""
        logprob_tree = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: jnp.sum(self.weight_prior.log_prob(x.reshape(-1))/self.temperature),
                                                                        params))

        return sum(logprob_tree)

    def log_likelihood(self, params, x, y):
        preds = self.apply(params, x).ravel()
        log_prob = jnp.sum(tfd.Normal(preds, self.temperature).log_prob(y))
        batch_size = x.shape[0]
        log_prob = (self.data_size / batch_size)*log_prob
        return log_prob

    def loss(self, params, x, y):
        logprob_prior = self.log_prior(params)
        logprob_likelihood = self.log_likelihood(params, x, y)
        return logprob_likelihood + logprob_prior

class DiscPosteriorInfo(NamedTuple):
    prior_logprob: float
    lik_logprob: float
    gradient: float

class BgBayesNN():
    def __init__(self, sgd_optim, sgld_optim, disc_sgd_optim, disc_sgld_optim,
                 temperature, sigma, data_size, hidden_sizes,
                 J, eta, mu,
                 act_fn, init_fn):
        self.hidden_sizes = hidden_sizes
        self.act_fn = act_fn
        self.sgd_optim = sgd_optim
        self.sgld_optim = sgld_optim
        self.optimiser = sgd_optim

        self.disc_optimiser = disc_sgd_optim
        self.disc_sgd_optim = disc_sgd_optim
        self.disc_sgld_optim = disc_sgld_optim

        self._forward = hk.transform(self._forward_fn)
        self.loss = jax.jit(self.loss)
        self.disc_loss = jax.jit(self.disc_loss)
        self.update = jax.jit(self.update)

        self.temperature = temperature
        self.sigma = sigma
        self.data_size = data_size
        self.add_noise = False
        self.J = J
        self.eta = eta
        self.mu = mu
        self.init_fn = init_fn

        # weight_decay = self.sigma*self.temperature
        # self.weight_prior = tfd.Normal(0, self.sigma)
        self.weight_prior = tfd.StudentT(df=2, loc=0, scale=self.sigma)
        # self.weight_prior = tfd.Laplace(loc=0, scale=sigma)

    def init(self, rng, x):
        gamma = tfd.Bernoulli(0.5*jnp.ones(x.shape[-1])).sample(seed=rng)*1.
        params = self._forward.init(rng, x, gamma, True)
        opt_state = self.optimiser.init(params)
        disc_opt_state = self.disc_optimiser.init(gamma)
        return params, gamma, opt_state, disc_opt_state

    def apply(self, params, gamma, x, is_training):
        return self._forward.apply(params, None, x, gamma, is_training).ravel()


    def loss(self, params, gamma, x, y):
        logprob_prior = self.log_prior(params)
        logprob_likelihood = self.log_likelihood(params, gamma, x, y)
        return (logprob_likelihood + logprob_prior)/self.temperature

    def disc_loss(self, gamma, params, x, y):
        prior_logprob = self.ising_prior(gamma)
        log_ll_prob = self.log_likelihood(params, gamma, x, y)
        return (prior_logprob + log_ll_prob)/self.temperature

    def update(self, key, params, gamma, opt_state, disc_opt_state, x, y):
        if self.add_noise:
            self.optimiser = self.sgld_optim
            self.disc_optimiser = self.disc_sgld_optim
        else:
            self.optimiser = self.sgd_optim
            self.disc_optimiser = self.disc_sgd_optim

        # contin_loss = lambda p: self.log_likelihood(p, gamma, x, y)

        grads = jax.grad(self.loss)(params, gamma, x, y)
        if self.add_noise:
            updates, opt_state = self.optimiser.update(grads, opt_state, key)
        else:
            updates, opt_state = self.optimiser.update(grads, opt_state)
        # updates, opt_state = self.optimiser.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        disc_grads = jax.grad(self.disc_loss)(gamma, params, x, y)
        gamma, disc_opt_state = self.disc_optimiser.update(key, gamma, disc_grads, disc_opt_state)
        # return params, gamma, opt_state, disc_opt_state, DiscPosteriorInfo(disc_prior_lp, disc_ll_lp, disc_grads)
        return params, gamma, opt_state, disc_opt_state

    def _forward_fn(self, x, gamma, is_training):
        if is_training:
            x = x @ jnp.diag(gamma)
        for hd in self.hidden_sizes:
            x = hk.Linear(hd, w_init=self.init_fn)(x)
            x = self.act_fn(x)

        x = hk.Linear(1)(x)
        return x

    def log_prior(self, params):
        """Computes the Gaussian prior log-density."""
        logprob_tree = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: jnp.sum(self.weight_prior.log_prob(x.reshape(-1))),
                                                                        params))

        return sum(logprob_tree)

    def log_likelihood(self, params, gamma, x, y):
        preds = self.apply(params, gamma, x, True).ravel()
        log_prob = jnp.sum(tfd.Normal(preds, 1.0).log_prob(y))
        batch_size = x.shape[0]
        log_prob = (self.data_size/batch_size)*log_prob

        # l2_regulariser = 0.5 * sum(
        #     jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))

        return log_prob

    def ising_prior(self, gamma):
        """Log probability of the Ising model - prior over the discrete variables"""
        # return (0.5*self.eta*(gamma.T @ self.J @ gamma) - self.mu*jnp.sum(gamma)) / self.temperature
        # t = 2*(gamma.T  @ gamma) - gamma + 1
        return (0.5*self.eta*(gamma.T @ self.J @ gamma) + self.mu*jnp.sum(gamma))
        # return (0.5*self.eta*(t.T @ self.J @ t) - self.mu*jnp.sum(gamma))



class BgBayesNN2():
    def __init__(self, sgd_optim, sgld_optim, disc_sgd_optim, disc_sgld_optim,
                 temperature, sigma, data_size, hidden_sizes,
                 J, eta, mu,
                 act_fn, init_fn):
        self.hidden_sizes = hidden_sizes
        self.act_fn = act_fn
        self.sgd_optim = sgd_optim
        self.sgld_optim = sgld_optim
        self.optimiser = sgd_optim

        self.disc_optimiser = disc_sgd_optim
        self.disc_sgd_optim = disc_sgd_optim
        self.disc_sgld_optim = disc_sgld_optim

        self._forward = hk.transform(self._forward_fn)
        self.loss = jax.jit(self.loss)
        self.disc_loss = jax.jit(self.disc_loss)
        self.update = jax.jit(self.update)

        self.temperature = temperature
        self.sigma = sigma
        self.data_size = data_size
        self.add_noise = False
        self.J = J
        self.eta = eta
        self.mu = mu
        self.init_fn = init_fn

        # weight_decay = self.sigma*self.temperature
        # self.weight_prior = tfd.Normal(0, self.sigma)
        self.weight_prior = tfd.StudentT(df=2, loc=0, scale=self.sigma)
        # self.weight_prior = tfd.Laplace(loc=0, scale=sigma)

    def init(self, rng, x):
        gamma = tfd.Bernoulli(0.5*jnp.ones(x.shape[-1])).sample(seed=rng)*1.
        params = self._forward.init(rng, x, gamma, True)
        opt_state = self.optimiser.init(params)
        disc_opt_state = self.disc_optimiser.init(gamma)
        return params, gamma, opt_state, disc_opt_state

    def apply(self, params, gamma, x, is_training):
        return self._forward.apply(params, None, x, gamma, is_training).ravel()


    def loss(self, params, gamma, x, y):
        logprob_prior = self.log_prior(params)
        logprob_likelihood = self.log_likelihood(params, gamma, x, y)
        return (logprob_likelihood + logprob_prior)/self.temperature

    def disc_loss(self, gamma, params, x, y):
        prior_logprob = self.ising_prior(gamma)
        log_ll_prob = self.log_likelihood(params, gamma, x, y)
        return (prior_logprob + log_ll_prob)/self.temperature

    def update(self, key, params, gamma, opt_state, disc_opt_state, x, y):
        if self.add_noise:
            self.optimiser = self.sgld_optim
            self.disc_optimiser = self.disc_sgld_optim
        else:
            self.optimiser = self.sgd_optim
            self.disc_optimiser = self.disc_sgd_optim

        # contin_loss = lambda p: self.log_likelihood(p, gamma, x, y)

        grads = jax.grad(self.loss)(params, gamma, x, y)
        if self.add_noise:
            updates, opt_state = self.optimiser.update(grads, opt_state, key)
        else:
            updates, opt_state = self.optimiser.update(grads, opt_state)
        # updates, opt_state = self.optimiser.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        disc_grads = jax.grad(self.disc_loss)(gamma, params, x, y)
        gamma, disc_opt_state = self.disc_optimiser.update(key, gamma, disc_grads, disc_opt_state)
        # return params, gamma, opt_state, disc_opt_state, DiscPosteriorInfo(disc_prior_lp, disc_ll_lp, disc_grads)
        return params, gamma, opt_state, disc_opt_state

    def _forward_fn(self, x, gamma, is_training):
        # if is_training:
        #     x = x @ jnp.diag(gamma)
        w = hk.get_parameter("w", [x.shape[-1], self.hidden_sizes[0]], init=lambda s, d: self.weight_prior.sample(
            seed=hk.next_rng_key(), sample_shape=s
        ))
        w_x = jnp.diag(gamma) @ w
        x = x @ w_x
        x = self.act_fn(x)
        for hd in self.hidden_sizes[1:]:
            x = hk.Linear(hd, w_init=self.init_fn)(x)
            x = self.act_fn(x)

        x = hk.Linear(1)(x)
        return x

    def log_prior(self, params):
        """Computes the Gaussian prior log-density."""
        logprob_tree = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: jnp.sum(self.weight_prior.log_prob(x.reshape(-1))),
                                                                        params))

        return sum(logprob_tree)

    def log_likelihood(self, params, gamma, x, y):
        preds = self.apply(params, gamma, x, True).ravel()
        log_prob = jnp.sum(tfd.Normal(preds, 1.0).log_prob(y))
        batch_size = x.shape[0]
        log_prob = (self.data_size/batch_size)*log_prob

        # l2_regulariser = 0.5 * sum(
        #     jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))

        return log_prob

    def ising_prior(self, gamma):
        """Log probability of the Ising model - prior over the discrete variables"""
        # return (0.5*self.eta*(gamma.T @ self.J @ gamma) - self.mu*jnp.sum(gamma)) / self.temperature
        # t = 2*(gamma.T  @ gamma) - gamma + 1
        return (0.5*self.eta*(gamma.T @ self.J @ gamma) + self.mu*jnp.sum(gamma))
        # return (0.5*self.eta*(t.T @ self.J @ t) - self.mu*jnp.sum(gamma))

class BgBayesClassifier(BgBayesNN):
    def __init__(self, sgd_optim, sgld_optim, disc_sgd_optim, disc_sgld_optim,
                 temperature, sigma, data_size, hidden_sizes,
                 J, eta, mu,
                 act_fn, init_fn):

        super().__init__(sgd_optim, sgld_optim, disc_sgd_optim, disc_sgld_optim,
                     temperature, sigma, data_size, hidden_sizes,
                     J, eta, mu,
                     act_fn, init_fn)

        self.weight_prior = tfd.StudentT(df=2, loc=0, scale=self.sigma)


    def log_likelihood(self, params, gamma, x, y):
        logits = self.apply(params, gamma, x, True).ravel()
        probs = jax.nn.sigmoid(logits)
        log_prob = jnp.sum(y*jnp.log(probs) + (1 - y)*jnp.log(1 - probs))
        batch_size = x.shape[0]
        log_prob = (self.data_size/batch_size)*log_prob
        return log_prob


# Author Abdulrahman S. Omar <hsamireh@gmail.com>
import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import nn_util
import tree_utils
from gibbs_sampler import *
import nn_models as models
import functools

class MixedSGMCMC(ClassifierMixin):

    def __init__(self, seed=1234, disc_lr=1e-5, contin_lr=1e-5, sigma=1.0, eta=0.1, mu=0.1,
                 alpha=0.95, batch_size=50, n_samples=10_000, n_warmup=1_000, n_chains=1,
                 num_cycles=10, lr_schedule="constant", temp=1.0, beta=0.5, layer_dims=None, output_dim=1,
                 thinning_interval=100, classifier=True):

        self.seed = seed
        self.disc_lr = disc_lr
        self.contin_lr = contin_lr
        self.sigma = sigma
        self.eta = eta
        self.mu = mu
        self.alpha = alpha
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.n_warmup = n_warmup
        self.n_chains = n_chains
        self.lr_schedule = lr_schedule
        self.num_cycles = num_cycles
        self.temp = temp
        self.beta = beta
        if layer_dims is None:
            self.layer_dims = [200]
        else:
            self.layer_dims = layer_dims

        self.output_dim = output_dim
        self.thinning_interval = thinning_interval

        self._classifier = classifier

        if classifier:
            self._estimator_type = "classifier"
        else:
            self._estimator_type = "regression"


    def fit(self, X, y, J=None, activation_fns=None):

        assert X.shape[0] == y.shape[0]
        self.rng_key_ = jax.random.PRNGKey(self.seed)
        data = Batch(X, y)
        data_size = X.shape[0]

        if self.lr_schedule not in ["constant", "cyclical", "exponential"]:
            raise ValueError(f"Unsupported Learning rate schedule: {self.lr_schedule}"
                             f", supported schedules are constant, cyclic, exponential")

        if J is None:
            raise ValueError(f"You need to provide the graph!")

        if activation_fns is None:
            raise ValueError(f"You must provide a list of activation functions with the same length as "
                        f"the layer dims")

        assert J.shape[0] == J.shape[1]
        p = J.shape[0]

        if self._classifier:
            loss_fn = cross_entropy_loss
        else:
            loss_fn = mse_loss


        log_likelihood_fn = functools.partial(mixed_loglikelihood_fn, loss_fn=loss_fn)

        mixed_net_fn = models.make_mixed_net_fn(layer_dims=self.layer_dims, activation_fns=activation_fns
                                                                            ,output_dim=self.output_dim)
        self.model_ = hk.without_apply_rng(hk.transform(mixed_net_fn))

        disc_logprior_fn = generate_disc_logprior_fn(J, self.mu, self.eta)
        contin_logprior_fn = generate_contin_logprior_fn(self.sigma)
        disc_grad_est_fn = generate_discrete_grad_estimator(self.model_, disc_logprior_fn, log_likelihood_fn ,data_size, self.temp)
        contin_grad_est_fn = geneate_mixed_contin_grad_estimator(self.model_ , contin_logprior_fn, log_likelihood_fn ,data_size, self.temp)

        disc_step_size_fn = nn_util.make_step_size_fn(self.disc_lr, self.lr_schedule, self.alpha, 
                                                            self.n_samples, self.num_cycles)

        contin_step_size_fn = nn_util.make_step_size_fn(self.contin_lr, self.lr_schedule, self.alpha, 
                                                            self.n_samples, self.num_cycles)

        key_samples, key_init = jax.random.split(self.rng_key_, 2)

        kernel = jax.jit(get_mixed_sgld_kernel(disc_grad_est_fn, contin_grad_est_fn,
                                               disc_step_size_fn, contin_step_size_fn))


        chain_keys = jax.random.split(key_init, self.n_chains)
        init_state = jax.vmap(make_init_mixed_state, in_axes=(0, None, None, None))(chain_keys,
                                                                    self.model_, data, self.batch_size)

        warmup_states, states = inference_loop_multiple_chains(key_samples, kernel, init_state, self.lr_schedule ,data, self.batch_size,
                                                                                self.n_samples, self.n_warmup,
                                                                                self.n_chains, num_cycles=self.num_cycles,
                                                                                beta=self.beta)

        if self.lr_schedule == "cyclical":
            start_dim = 3
        else:
            start_dim = 2

        discrete_position = tree_utils.combine_dims(states.discrete_position, start_dim)
        contin_position = tree_utils.combine_dims(states.contin_position, start_dim)
        # disc_precond = tree_utils.combine_dims(states.disc_precond, start_dim)
        # contin_precond = tree_utils.combine_dims(states.contin_precond, start_dim)

        # discrete_position = discrete_position[::self.thinning_interval,]
        # contin_position = jax.tree_util.tree_map(lambda pos: pos[::self.thinning_interval,], contin_position)
        self.states_ = MixedState(states.count, discrete_position, contin_position)


        if self.lr_schedule != "cyclical":
            # TODO: Remove keeping warmup samples. It just for diagnostics
            discrete_position = tree_utils.combine_dims(warmup_states.discrete_position, 2)
            contin_position = tree_utils.combine_dims(warmup_states.contin_position, 2)
            disc_precond = tree_utils.combine_dims(warmup_states.disc_precond, 2)
            contin_precond = tree_utils.combine_dims(warmup_states.contin_precond, 2)

            self.warmup_states_ = MixedState(warmup_states.count, discrete_position, contin_position)

        return self


    def predict(self, X):
        if self.model_ is None:
            raise ValueError("Model is not fitted yet. Call the fit method")
        if self._classifier:
            pred_fn = lambda p, g: get_mixed_model_pred(self.model_, p, X, g)
        else:
            pred_fn = lambda p, g: self.model_.apply(p, X, g).ravel()

        preds = jax.vmap(pred_fn)(self.states_.contin_position, self.states_.discrete_position)

        return jnp.mean(preds, axis=0).ravel() #ensemble predictions

    def score(self, X, y):
        preds = self.predict(X)

        if self._classifier:
            return roc_auc_score(y, preds)
        else:
            return jnp.mean((y - preds)**2)

    def get_params(self, deep=True):
        return {"seed": self.seed, "disc_lr": self.disc_lr, "contin_lr": self.contin_lr,
                "sigma": self.sigma, "eta": self.eta, "mu": self.mu, "alpha": self.alpha,
                "batch_size": self.batch_size, "n_samples": self.n_samples, "n_warmup": self.n_warmup,
                "n_chains": self.n_chains, "lr_schedule": self.lr_schedule, "num_cycles": self.num_cycles,
                "temp": self.temp, "layer_dims": self.layer_dims, "output_dim": self.output_dim}

    def set_params(self, **params):
        for key, param in params.items():
            setattr(self, key, param)

        return self

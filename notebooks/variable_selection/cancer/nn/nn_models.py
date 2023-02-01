# Author Abdulrahman S. Omar <hsamireh@gmail.com>

import jax
import jax.numpy as jnp
import haiku as hk
import tensorflow_probability.substrates.jax as tfp
import optax

tfd = tfp.distributions

def make_mixed_net_fn(*, layer_dims, activation_fns ,output_dim):
    assert len(layer_dims) == len(activation_fns)
    w_init = hk.initializers.VarianceScaling()
    def forward(x, gamma):
        x = hk.Flatten()(x)
        x = jnp.dot(x, jnp.diag(gamma))
        for layer_dim, activation_fn in zip(layer_dims, activation_fns):
            x = hk.Linear(layer_dim, w_init=w_init)(x)
            x = get_act_fn(activation_fn)(x)
        x = hk.Linear(output_dim)(x)
        return x

    return forward

def make_net_fn(*, layer_dims, activation_fns, output_dim):
    assert len(layer_dims) == len(activation_fns)
    w_init = hk.initializers.VarianceScaling()
    def forward(x):
        x = hk.Flatten()(x)
        for layer_dim, activation_fn in zip(layer_dims, activation_fns):
            x = hk.Linear(layer_dim, w_init=w_init)(x)
            x = get_act_fn(activation_fn)(x)
        x = hk.Linear(output_dim)(x)
        return x

    return forward

def get_act_fn(name):
    if name == "relu":
        return jax.nn.relu
    if name == "swish":
        return jax.nn.swish
    if name == "tanh":
        return jax.nn.tanh
    if name == "sigmoid":
        return jax.nn.sigmoid
    if name == "celu":
        return jax.nn.celu
    if name == "relu6":
        return jax.nn.relu6
    if name == "glu":
        return jax.nn.glu
    if name == "elu":
        return jax.nn.elu
    if name == "leaky_relu":
        return jax.nn.leaky_relu
    if name == "log_sigmoid":
        return jax.nn.log_sigmoid

    return ValueError(f"Unknown activation function: {name}")


def make_sgld_net_fn(*, layer_dims, activation_fns, output_dim):
    w_init = hk.initializers.RandomNormal(0., 1.)
    def forward(x):
        x = hk.Flatten()(x)
        for layer_dim, activation_fn in zip(layer_dims, activation_fns):
            x = hk.Linear(layer_dim, w_init=w_init)(x)
            x = activation_fn(x)
        x = hk.Linear(output_dim)(x)
        return x

    return forward

class TrainingState(NamedTuple):
    params: hk.Params
    avg_params: hk.Params
    opt_state: optax.OptState
    net_state: hk.State

class ResNetBlock(hk.Module):
    def __init__(self, act_fn, hidden_dim, skip_dim, dropout_rate, name=None):
        super().__init__(name)
        self.act_fn = act_fn
        self.hidden_dim = hidden_dim
        self.skip_dim = skip_dim
        self.dropout_rate = dropout_rate

    def __call__(self, x, is_training):
        key = hk.next_rng_key()
        init_fn = hk.initializers.VarianceScaling()
        z = hk.BatchNorm(True, True, 0.9)(x, is_training)
        z = self.act_fn(z)
        z = hk.Linear(self.hidden_dim, w_init=init_fn)(z)
        z = self.act_fn(z)
        if is_training:
            z = hk.dropout(key, self.dropout_rate, z)
        z = hk.Linear(self.skip_dim, w_init=init_fn)(z)
        z = hk.BatchNorm(True, True, 0.9)(z, is_training)
        z = self.act_fn(z)
        if is_training:
            z = hk.dropout(key, self.dropout_rate, z)

        return z

class ResNetBlock2(hk.Module):

    def __init__(self, act_fn, hidden_dims, dropout_rate, name=None):
        super().__init__(name)
        self.act_fn = act_fn
        self.dropout_rate = dropout_rate
        self.init_fn = hk.initializers.VarianceScaling()
        self.layers = [hk.Linear(dim, w_init=self.init_fn, b_init=self.init_fn) for dim in hidden_dims]


    def __call__(self, x, is_training):
        key = hk.next_rng_key()
        for layer in self.layers:
            x = hk.BatchNorm(False, False, 0.9)(x, is_training)
            x = layer(x)
            x = self.act_fn(x)
            x = hk.BatchNorm(False, False, 0.9)(x, is_training)
            if is_training:
                x = hk.dropout(key, self.dropout_rate, x)
        return x


class ResNet:
    def __init__(self, sgd_optim, sgld_optim, num_blocks, hidden_dims ,dropout_rate=0.5, act_fn=jax.nn.relu):
        self.num_blocks = num_blocks
        self.hidden_dims = hidden_dims
        self.act_fn = act_fn
        self.sgd_optim = sgd_optim
        self.sgld_optim = sgld_optim
        self.optimiser = sgd_optim
        self._forward = hk.transform_with_state(self._forward_fn)
        self.loss = jax.jit(self.loss)
        self.update = jax.jit(self.update)
        self.add_noise = False
        self.dropout_rate = dropout_rate

    def init(self, rng, x):
        params, net_state = self._forward.init(rng, x, is_training=True)
        opt_state = self.optimiser.init(params)
        return TrainingState(params, params, opt_state, net_state)

    def apply(self, params, net_state, x, key, is_training=True):
        return self._forward.apply(params, net_state, key, x, is_training)


    def update(self, key, train_state, x, y):
        if self.add_noise:
            self.optimiser = self.sgld_optim
        else:
            self.optimiser = self.sgd_optim

        params, avg_params, opt_state, net_state = train_state
        grads, net_state = jax.grad(self.loss, has_aux=True)(params, net_state, key, x, y)
        updates, opt_state = self.optimiser.update(grads, opt_state, key)
        params = optax.apply_updates(params, updates)
        avg_params = optax.incremental_update(params, avg_params, step_size=0.001)
        return TrainingState(params, avg_params, opt_state, net_state)

    def _forward_fn(self, x, is_training):
        init_fn = hk.initializers.VarianceScaling()
        x = hk.Linear(self.hidden_dims[-1], w_init=init_fn, b_init=init_fn)(x)
        for _ in range(self.num_blocks):
            z = x
            z = ResNetBlock2(self.act_fn, self.hidden_dims, self.dropout_rate)(z, is_training)
            x = z + x # add skip connection

        x = hk.BatchNorm(False, False, 0.9)(x, is_training)
        x = self.act_fn(x)
        x = hk.Linear(2)(x)

        return x

    def loss(self, params, net_state, key, x, y):
        preds, state = self.apply(params, net_state, x, key, is_training=True)
        preds_mean, preds_std = jnp.split(preds, [1], axis=-1)
        preds_std = jax.nn.softplus(preds_std)
        preds_mean = preds_mean.squeeze()
        se = (y - preds_mean)**2
        log_likelihood = (-0.5*se / (preds_std**2) -
                          0.5*jnp.log((preds_std**2)*2*jnp.pi))
        log_likelihood = jnp.sum(log_likelihood)
        # preds = preds.squeeze()
        # l2_loss = jnp.mean(optax.l2_loss(y, preds))
        # l2_reg =  l2_regulariser = 0.5 * sum(
        #         jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))

        return -log_likelihood, state
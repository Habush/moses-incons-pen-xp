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
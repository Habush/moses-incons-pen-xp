# Author Abdulrahman S. Omar <hsamireh@gmail.com>

import jax
import jax.numpy as jnp
import haiku as hk

def make_mixed_net_fn(*, layer_dims, activation_fns ,output_dim):
    assert len(layer_dims) == len(activation_fns)
    w_init = hk.initializers.RandomNormal(0., 1.)
    def forward(x, gamma):
        x = hk.Flatten()(x)
        x = jnp.dot(x, jnp.diag(gamma))
        for layer_dim, activation_fn in zip(layer_dims, activation_fns):
            x = hk.Linear(layer_dim, w_init=w_init)(x)
            x = get_activation_fn(activation_fn)(x)
        x = hk.Linear(output_dim)(x)
        return x

    return forward

def get_activation_fn(fn_name):
    if fn_name == "tanh":
        return jax.nn.tanh
    if fn_name == "sigmoid":
        return jax.nn.sigmoid
    if fn_name == "relu":
        return jax.nn.relu
    if fn_name == "selu":
        return jax.nn.selu

    raise ValueError(f"Unknown activation function: {fn_name}. Options are [tanh, relu, selu, sigmoid]")

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
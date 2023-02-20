import torch
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
from bnn_models import *
from nn_util import roc_auc_score
from resnet_models import *
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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


def init_bnn_model(seed, train_loader, epochs, lr_0, num_cycles, temp, sigma, hidden_sizes, act_fn):
    torch.manual_seed(seed)
    num_batches = len(train_loader)
    total_steps = num_batches*epochs
    data_size = train_loader.dataset.data.shape[0]
    step_size_fn = make_cyclical_lr_fn(lr_0, total_steps, num_cycles)
    sgd_optim = sgd_gradient_update(step_size_fn, momentum_decay=0.9, preconditioner=get_rmsprop_preconditioner())
    sgld_optim = sgld_gradient_update(step_size_fn, momentum_decay=0.9, preconditioner=get_rmsprop_preconditioner())

    model = BayesNN(sgd_optim, sgld_optim,
                    temp, sigma, data_size, hidden_sizes, act_fn)

    return model


def train_bnn_model(seed, train_loader, epochs, num_cycles, beta, lr_0,
                    hidden_sizes, temp, sigma, act_fn=jax.nn.relu):

    rng_key = jax.random.PRNGKey(seed)
    model = init_bnn_model(seed, train_loader, epochs, lr_0, num_cycles, temp, sigma, hidden_sizes, act_fn)

    # cycle_len = epochs // num_cycles
    num_batches = len(train_loader)
    M = (epochs*num_batches) // num_cycles
    init_params, init_opt_state = model.init(rng_key, next(iter(train_loader))[0])


    states = []
    params, opt_state = init_params, init_opt_state
    step = 0
    key = rng_key
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            _, key = jax.random.split(key, 2)
            rk = (step % M) / M
            params, opt_state = model.update(key, params, opt_state, batch_x, batch_y)
            if rk > beta:
                model.add_noise = True
                states.append(params)
            else:
                model.add_noise = False
            step += 1

    return model, states

def init_bg_bnn_model(seed, train_loader, epochs, lr_0, disc_lr_0, num_cycles, temp, sigma, hidden_sizes, J, eta, mu, act_fn, init_fn, classifier):
    torch.manual_seed(seed)
    num_batches = len(train_loader)
    data_size = train_loader.dataset.data.shape[0]
    total_steps = num_batches*epochs
    step_size_fn = make_cyclical_lr_fn(lr_0, total_steps, num_cycles)
    disc_step_size_fn = make_cyclical_lr_fn(disc_lr_0, total_steps, num_cycles)

    sgd_optim = sgd_gradient_update(step_size_fn, momentum_decay=0.9, preconditioner=get_rmsprop_preconditioner())
    sgld_optim = sgld_gradient_update(step_size_fn, momentum_decay=0.9, preconditioner=get_rmsprop_preconditioner())
    # sgd_optim = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(step_size_fn))
    # sgld_optim = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(step_size_fn))
    #TODO change this
    disc_sgd_optim = disc_sgld_gradient_update(disc_step_size_fn, momentum_decay=0.9, preconditioner=get_rmsprop_preconditioner())
    disc_sgld_optim = disc_sgld_gradient_update(disc_step_size_fn, momentum_decay=0.9, preconditioner=get_rmsprop_preconditioner())

    if classifier:
        model = BgBayesClassifier(sgd_optim, sgld_optim, disc_sgd_optim, disc_sgld_optim,
                      temp, sigma, data_size, hidden_sizes,
                      J, eta, mu, act_fn, init_fn)

    else:
        model = BgBayesNN(sgd_optim, sgld_optim, disc_sgd_optim, disc_sgld_optim,
                          temp, sigma, data_size, hidden_sizes,
                          J, eta, mu, act_fn, init_fn)

    return model


def train_bg_bnn_model(seed, train_loader, epochs, num_cycles, beta, m, lr_0, disc_lr_0,
                       hidden_sizes, temp, sigma, eta, mu, J, act_fn_name,
                       show_pgbar=True, classifier=False):

    rng_key = jax.random.PRNGKey(seed)
    act_fn = get_act_fn(act_fn_name)
    init_fn = hk.initializers.VarianceScaling()

    model = init_bg_bnn_model(seed, train_loader, epochs, lr_0, disc_lr_0, num_cycles, temp, sigma, hidden_sizes, J, eta, mu, act_fn, init_fn, classifier)

    num_batches = len(train_loader)
    M = (epochs*num_batches) // num_cycles
    cycle_len = epochs // num_cycles
    init_params, init_gamma, init_opt_state, init_disc_opt_state = model.init(rng_key, next(iter(train_loader))[0])

    states = []
    disc_states = []

    params, gamma, opt_state, disc_opt_state = init_params, init_gamma, init_opt_state, init_disc_opt_state
    step = 0
    key = rng_key

    if show_pgbar:
        pgbar = tqdm(range(epochs))
    else:
        pgbar = range(epochs)

    for epoch in pgbar:
        for batch_x, batch_y in train_loader:
            _, key = jax.random.split(key, 2)
            rk = (step % M) / M
            params, gamma, opt_state, disc_opt_state = model.update(key, params, gamma, opt_state, disc_opt_state, batch_x, batch_y)
            if rk > beta:
                model.add_noise = True
            else:
                model.add_noise = False

            step += 1
        if epoch != 0 and (epoch % cycle_len) + 1 > (cycle_len - m):
            states.append(params)
            disc_states.append(gamma)

    return model, states, disc_states


def train_mlp_model(seed, train_loader, epochs, lr_0, hidden_sizes, act_fn_name, show_pgbar=True):
    rng_key = jax.random.PRNGKey(seed)
    act_fn = get_act_fn(act_fn_name)
    init_fn = hk.initializers.VarianceScaling()
    optim = optax.chain(optax.scale_by_adam(), optax.scale(-lr_0)) # since we're minimizing a loss

    model = MLP(optim, hidden_sizes, init_fn, act_fn)

    init_params, init_opt_state = model.init(rng_key, next(iter(train_loader))[0])
    params, opt_state = init_params, init_opt_state
    if show_pgbar:
        pgbar = tqdm(range(epochs))
    else:
        pgbar = range(epochs)
    for _ in pgbar:
        for x, y in train_loader:
            params, opt_state = model.update(params, opt_state, x, y)

    return model, params

def eval_bg_bnn_model(model, X, y, params, gammas, is_training, classifier=False):

    if isinstance(params, list):
        y_preds = np.zeros((len(params), len(y)))
        for i, (param, gamma) in enumerate(zip(params, gammas)):
            preds = model.apply(param, gamma, X, is_training).ravel()
            if classifier:
                y_preds[i] = jax.nn.sigmoid(preds)
            else:
                y_preds[i] = preds

        y_preds = np.mean(y_preds, axis=0)
        if classifier:
            score = roc_auc_score(y, y_preds)
        else:
            score = jnp.sqrt(jnp.mean((y - y_preds)**2))

    else:
        y_preds = model.apply(params, gammas, X, is_training).ravel()
        if classifier:
            y_preds = jax.nn.sigmoid(y_preds)
            score = roc_auc_score(y, y_preds)
        else:
            score = jnp.sqrt(jnp.mean((y - y_preds)**2))

    return score

def score_bg_bnn_model(model, X, y, params, gammas, is_training, classifier=False):
    if isinstance(params, list):
        y_preds = np.zeros((len(params), len(y)))
        for i, (param, gamma) in enumerate(zip(params, gammas)):
            preds = model.apply(param, gamma, X, is_training).ravel()
            if classifier:
                y_preds[i] = jax.nn.sigmoid(preds)
            else:
                y_preds[i] = preds

        y_preds = np.mean(y_preds, axis=0)
        if classifier:
            score = roc_auc_score(y, y_preds)
            acc = accuracy_score(y, y_preds > 0.5)
            return score, acc
        else:
            score = jnp.sqrt(jnp.mean((y - y_preds)**2))
            if np.isfinite(y_preds).all():
                r2 = r2_score(y, y_preds)
            else:
                r2 = np.nan
            return score, r2
    else:
        y_preds = model.apply(params, gammas, X, is_training).ravel()
        if classifier:
            y_preds = jax.nn.sigmoid(y_preds)
            score = roc_auc_score(y, y_preds)
            acc = accuracy_score(y, y_preds > 0.5)
            return score, acc
        else:
            score = jnp.sqrt(jnp.mean((y - y_preds)**2))
            if np.isfinite(y_preds).all():
                r2 = r2_score(y, y_preds)
            else:
                r2 = np.nan
            return score, r2

def eval_per_model_score_bg(model, X, y, params, gammas):
    scores = []

    for param, gamma in zip(params, gammas):
        preds = model.apply(param, gamma, X).ravel()
        # preds_mean = preds[::2]
        rmse = jnp.sqrt(jnp.mean((y - preds)**2))
        scores.append(rmse)



    return np.array(scores)


def train_nn_model(rng_key, data_loader, epochs, num_cycles, lr_0,
                   block_type, num_blocks, hidden_sizes, init_fn, weight_decay,
                   act_fn_name, dropout_rate ,show_pgbar=True):


    act_fn = get_act_fn(act_fn_name)
    total_steps = len(data_loader)*epochs

    schedule_fn = make_cyclical_lr_fn(lr_0, total_steps, num_cycles)
    optim = optax.chain(optax.scale_by_adam(), optax.add_decayed_weights(weight_decay),
                        optax.scale_by_schedule(schedule_fn), optax.scale(-1.0))
    if block_type == "ResNet":
        block_class = ResNetBlock
    else:
        block_class = PreActResNetBlock

    model = ResNet(block_class, num_blocks, hidden_sizes, optim, init_fn, act_fn, dropout_rate)

    cycle_len = epochs // num_cycles
    init_state = model.init(rng_key, next(iter(data_loader))[0])

    state = init_state

    # print(f"Total iterations: {epochs*num_batches}, Num Batches: {num_batches}, Cycle Len: {M}")
    states = []
    val_losses = []
    step = 0
    key = rng_key

    if show_pgbar:
        pgbar = tqdm(range(epochs))
    else:
        pgbar = range(epochs)

    for epoch in pgbar:
        for batch_x, batch_y in data_loader:
            _, key = jax.random.split(key, 2)
            state = model.update(key, state, batch_x, batch_y)
            step += 1

        if epoch != 0 and ((epoch + 1) % cycle_len == 0): # take snapshot
            states.append(state)


    return model, states, val_losses



def train_resnet_bg_model(rng_key, data_loader, epochs, num_cycles, m, lr_0, disc_lr_0,
                   block_type, num_blocks, hidden_sizes, init_fn, weight_decay,
                   act_fn_name, dropout_rate, eta, mu, J, show_pgbar=True):


    act_fn = get_act_fn(act_fn_name)
    total_steps = len(data_loader)*epochs

    step_size_fn = make_cyclical_lr_fn(lr_0, total_steps, num_cycles)
    optim = optax.chain(optax.scale_by_adam(), optax.add_decayed_weights(weight_decay),
                        optax.scale_by_schedule(step_size_fn), optax.scale(-1.0))

    disc_schedule_fn = make_cyclical_lr_fn(disc_lr_0, total_steps, num_cycles)
    disc_optim = disc_sgld_gradient_update(disc_schedule_fn, momentum_decay=0.9, preconditioner=get_rmsprop_preconditioner())

    if block_type == "ResNet":
        block_class = ResNetBlock
    else:
        block_class = PreActResNetBlock

    model = BgResNet(block_class, num_blocks, hidden_sizes, optim, disc_optim,
                              init_fn, act_fn, dropout_rate,
                              eta, mu, J, data_loader.dataset.data.shape[0])

    cycle_len = epochs // num_cycles
    init_state = model.init(rng_key, next(iter(data_loader))[0])

    state = init_state

    # print(f"Total iterations: {epochs*num_batches}, Num Batches: {num_batches}, Cycle Len: {M}")
    states = []
    val_losses = []
    step = 0
    key = rng_key

    if show_pgbar:
        pgbar = tqdm(range(epochs))
    else:
        pgbar = range(epochs)

    for epoch in pgbar:
        for batch_x, batch_y in data_loader:
            _, key = jax.random.split(key, 2)
            state = model.update(key, state, batch_x, batch_y)
            step += 1

        if epoch != 0 and (epoch % cycle_len) + 1 > (cycle_len - m): # take snapshot
            states.append(state)


    return model, states, val_losses


def eval_nn_model(key, model, x, y, states):

    if isinstance(states, list):
        y_preds = np.zeros((len(states), len(y)))
        for i, state in enumerate(states):
            preds, _ = model.apply(state, x, key, False)
            y_preds[i] = preds.squeeze()

        y_preds = np.mean(y_preds, axis=0)
        rmse = jnp.sqrt(jnp.mean((y - y_preds)**2))
    else:
        y_preds, _ = model.apply(states, x, key, False)
        rmse = jnp.sqrt(jnp.mean((y - y_preds.squeeze())**2))

    return rmse

def eval_mlp_model(model, x, y, state):
    y_preds = model.apply(state, x)
    return jnp.sqrt(jnp.mean((y - y_preds)**2))

def eval_resnet_bg_model(key, model, x, y, states, is_training=False, eval_r2=False):

    if isinstance(states, list):
        y_preds = np.zeros((len(states), len(y)))
        for i, state in enumerate(states):
            preds, _ = model.apply(state, x, key, is_training)
            y_preds[i] = preds.squeeze()

        y_preds = np.mean(y_preds, axis=0)
        rmse = jnp.sqrt(jnp.mean((y - y_preds)**2))
    else:
        y_preds, _ = model.apply(states, x, key, is_training)
        rmse = jnp.sqrt(jnp.mean((y - y_preds.squeeze())**2))


    if eval_r2:
        r2 = r2_score(y, y_preds)
        return rmse, r2
    return rmse


def train_rf_model(seed, X, y, train_idxs, val_idxs, classifier=False):

    # cv = KFold(n_splits=5)
    cv = [(train_idxs, val_idxs) for _ in range(5)]
    param_grid = {
        'max_depth': [80, 100, 120],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 500, 1000]
    }

    if classifier:
        rf_model = RandomForestClassifier(random_state=seed, max_samples=1.0)
        scoring = "roc_auc"
    else:
        rf_model = RandomForestRegressor(random_state=seed, max_samples=1.0)
        scoring = "neg_root_mean_squared_error"
    grid_cv = GridSearchCV(estimator = rf_model, param_grid = param_grid,
                           cv = cv, n_jobs = -1, verbose = 0, scoring=scoring).fit(X, y)

    if classifier:
        rf_model = RandomForestClassifier(random_state=seed, max_samples=1.0, **grid_cv.best_params_)
    else:
        rf_model = RandomForestRegressor(random_state=seed, max_samples=1.0, **grid_cv.best_params_)

    rf_model.fit(X, y)

    return rf_model

def eval_rf_model(model, X, y, classifier=False):
    if classifier:
        y_preds = model.predict_proba(X)[:,1]
        auc = roc_auc_score(y, y_preds)
        acc = accuracy_score(y, y_preds > 0.5)
        return auc, acc
    else:
        y_preds = model.predict(X)
        rmse = jnp.sqrt(jnp.mean((y - y_preds)**2))
        r2 = r2_score(y, y_preds)
        return rmse, r2

def zero_out_score(model, X, y, states, disc_states, m, lst, is_training):
    feat_idxs = lst[:m]
    mask = np.zeros(X.shape[1])
    mask[feat_idxs] = 1.0
    X_mask = X @ np.diag(mask)
    rmse, _ = score_bg_bnn_model(model, X_mask, y, states, disc_states, is_training)
    return rmse

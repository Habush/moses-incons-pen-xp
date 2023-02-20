from data_utils import NumpyLoader, NumpyData
from train_utils import *
import optuna
import torch

def objective_bg_bnn(trial, seed, x_train, x_val, y_train, y_val, J, beta,
              hidden_sizes, act_fn, bg=True, classifier=False):

    # lr_0 =  trial.suggest_categorical("lr_0", [1e-3, 1e-2, 1e-1])
    lr_0 = 1e-3
    disc_lr_0 = trial.suggest_categorical("disc_lr_0", [0.1, 0.5])
    # lr_0, disc_lr_0 = 1e-3, 0.5
    num_cycles = trial.suggest_categorical("num_cycles", [10, 20, 30, 40, 50])
    temp = trial.suggest_categorical("temp", [1e-3, 1e-2, 1e-1, 1.])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    # epochs = trial.suggest_categorical("epochs", [50, 100, 150, 200, 250, 300])
    epochs = 500
    m = 1
    sigma = trial.suggest_categorical("sigma", [1e-1, 0.5, 1.])
    if bg:
        eta = trial.suggest_float("eta", -1e2, 1e2)
        # eta_sign = trial.suggest_categorical("eta_sign", [-1, 1])
        # signed_eta = eta_sign*eta
    else:
        eta = 1.0

    mu = trial.suggest_float("mu", 1.0, 1e2)

    torch.manual_seed(seed)
    train_loader = NumpyLoader(NumpyData(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    bg_bnn_model, states, disc_states = train_bg_bnn_model(seed, train_loader, epochs, num_cycles, beta, m, lr_0, disc_lr_0,
                                                           hidden_sizes, temp, sigma, eta, mu, J, act_fn,
                                                           show_pgbar=False, classifier=classifier)

    # val_losses = eval_per_model_score_bg(bg_bnn_model, x_val, y_val, states, disc_states)
    # model_idxs = jnp.argsort(val_losses).squeeze()[:M]
    # states_sel, disc_states_sel = list(itemgetter(*model_idxs)(states)), list(itemgetter(*model_idxs)(disc_states))
    if classifier:
        score = eval_bg_bnn_model(bg_bnn_model, x_val, y_val, states, disc_states, True, True)
        return 1 - score
    else:
        rmse = eval_bg_bnn_model(bg_bnn_model, x_val, y_val, states, disc_states, True)

        return rmse


def objective_resnet(trial, seed, x_train, x_val, y_train, y_val,
                     num_blocks, hidden_sizes, init_fn, act_fn):

    lr_0 = trial.suggest_categorical("lr_0", [1e-3, 5e-3, 1e-2, 1e-1])
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1., log=True)
    # dropout_rate = trial.suggest_categorical("dropout_rate", [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    dropout_rate = 0.0
    num_cycles = trial.suggest_categorical("num_cycles", [2, 4, 6, 8, 10])
    block_type = trial.suggest_categorical("block_type", ["ResNet", "PreActResNet"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_categorical("epochs", [50, 100, 150, 200, 250, 300])
    rng = jax.random.PRNGKey(seed)

    torch.manual_seed(seed)
    train_loader = NumpyLoader(NumpyData(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    bnn_model, state, _ = train_nn_model(rng, train_loader, epochs, num_cycles, lr_0,
                                         block_type, num_blocks, hidden_sizes,
                                         init_fn, weight_decay, act_fn, dropout_rate ,show_pgbar=False)

    rmse = eval_nn_model(rng, bnn_model, x_val, y_val, state)

    return rmse


def objective_resnet_bg(trial, seed, x_train, x_val, y_train, y_val,
                     init_fn, act_fn, J, bg=True):

    lr_0 = trial.suggest_categorical("lr_0", [1e-3,1e-2, 1e-1])
    disc_lr_0 = trial.suggest_categorical("disc_lr_0", [1e-2, 1e-1, 0.5])
    # lr_0 = 1e-2
    # disc_lr_0 = 0.5
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1., log=True)
    # dropout_rate = trial.suggest_categorical("dropout_rate", [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    dropout_rate = 0.0
    # num_cycles = trial.suggest_categorical("num_cycles", [10, 20, 30, 50])
    num_cycles = 20
    block_type = trial.suggest_categorical("block_type", ["ResNet", "PreActResNet"])
    # batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    batch_size = 32
    # epochs = trial.suggest_categorical("epochs", [200, 300, 400, 50])
    epochs = 200
    num_blocks = trial.suggest_categorical("num_blocks", [1, 2, 3, 4])
    block_size = trial.suggest_categorical("block_size", [32, 64, 128, 256])

    blocks = [1 for _ in range(num_blocks)]
    hidden_sizes = [block_size for _ in range(num_blocks)]

    m = 1

    if bg:
        eta = trial.suggest_float("eta", -1e2, 1e2)
    else:
        eta = 1.0
    mu = trial.suggest_float("mu", 1.0, 1e2)


    rng = jax.random.PRNGKey(seed)
    torch.manual_seed(seed)
    train_loader = NumpyLoader(NumpyData(x_train, y_train), batch_size=batch_size,
                               shuffle=True, drop_last=True)
    bnn_model, state, _ = train_resnet_bg_model(rng, train_loader, epochs, num_cycles, m, lr_0, disc_lr_0,
                                         block_type, blocks, hidden_sizes,
                                         init_fn, weight_decay, act_fn, dropout_rate,
                                         eta, mu, J, show_pgbar=False)

    rmse = eval_resnet_bg_model(rng, bnn_model, x_val, y_val, state)

    return rmse
import jax
import jax.numpy as jnp
import numpy as np
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import optax
from typing import NamedTuple, Any
import logging
from logging import handlers
import tensorflow_probability.substrates.jax as tfp
import scipy.stats as stats
import haiku as hk
from torch.utils import data
import tree_utils
from tqdm import tqdm

PRNGKey = Any

class Batch(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray


def recall(y_true, y_pred):
    true_positives = jnp.sum(jnp.round(jnp.clip(y_true * y_pred, 0, 1)))
    possible_positives = jnp.sum(jnp.round(jnp.clip(y_true, 0, 1)))
    return (true_positives / (possible_positives + 1e-12))

def precision(y_true, y_pred):
    true_positives = jnp.sum(jnp.round(jnp.clip(y_true * y_pred, 0, 1)))
    predicted_positives = jnp.sum(jnp.round(jnp.clip(y_pred, 0, 1)))
    return (true_positives / (predicted_positives + 1e-12))

def f1_score(y_true, y_pred):
    rec = recall(y_true, y_pred)
    prec = precision(y_true, y_pred)
    return 2*((prec*rec) / (prec + rec + 1e-12))



def _binary_clf_curve(y_true, y_score):
    # source https://ethen8181.github.io/machine-learning/model_selection/auc/auc.html
    """
    Calculate true and false positives per binary classification
    threshold (can be used for roc curve or precision/recall curve);
    the calcuation makes the assumption that the positive case
    will always be labeled as 1

    Parameters
    ----------
    y_true : 1d ndarray, shape = [n_samples]
        True targets/labels of binary classification

    y_score : 1d ndarray, shape = [n_samples]
        Estimated probabilities or scores

    Returns
    -------
    tps : 1d ndarray
        True positives counts, index i records the number
        of positive samples that got assigned a
        score >= thresholds[i].
        The total number of positive samples is equal to
        tps[-1] (thus false negatives are given by tps[-1] - tps)

    fps : 1d ndarray
        False positives counts, index i records the number
        of negative samples that got assigned a
        score >= thresholds[i].
        The total number of negative samples is equal to
        fps[-1] (thus true negatives are given by fps[-1] - fps)

    thresholds : 1d ndarray
        Predicted score sorted in decreasing order

    References
    ----------
    Github: scikit-learn _binary_clf_curve
    - https://github.com/scikit-learn/scikit-learn/blob/ab93d65/sklearn/metrics/ranking.py#L263
    """

    # sort predicted scores in descending order
    # and also reorder corresponding truth values
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically consists of tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve
    distinct_indices = np.where(np.diff(y_score))[0]
    end = np.array([y_true.size - 1])
    threshold_indices = np.hstack((distinct_indices, end))

    thresholds = y_score[threshold_indices]
    tps = np.cumsum(y_true)[threshold_indices]

    # (1 + threshold_indices) = the number of positives
    # at each index, thus number of data points minus true
    # positives = false positives
    fps = (1 + threshold_indices) - tps
    return tps, fps, thresholds



def roc_auc_score(y_true, y_score):
    #  source https://ethen8181.github.io/machine-learning/model_selection/auc/auc.html
    """
    Compute Area Under the Curve (AUC) from prediction scores

    Parameters
    ----------
    y_true : 1d ndarray, shape = [n_samples]
        True targets/labels of binary classification

    y_score : 1d ndarray, shape = [n_samples]
        Estimated probabilities or scores

    Returns
    -------
    auc : float
    """

    # ensure the target is binary
    # if np.unique(y_true).size != 2:
    #     raise ValueError('Only two class should be present in y_true. ROC AUC score '
    #                      'is not defined in that case.')

    tps, fps, _ = _binary_clf_curve(y_true, y_score)

    # convert count to rate
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]

    # compute AUC using the trapezoidal rule;
    # appending an extra 0 is just to ensure the length matches
    zero = np.array([0])
    tpr_diff = np.hstack((np.diff(tpr), zero))
    fpr_diff = np.hstack((np.diff(fpr), zero))
    auc = np.dot(tpr, fpr_diff) + np.dot(tpr_diff, fpr_diff) / 2
    return auc


def compute_model_accuracy(param, gamma, model, X, y):
    logits = model.apply(param, X, gamma)
    pred = (logits > 0).astype(jnp.float32)
    acc = jnp.mean(pred == y)
    return acc

def model_prediction(model, params, x, gamma):
    logits = model.apply(params, x, gamma).ravel()
    return (logits > 0).astype(jnp.float32)

def posterior_prob_predict(model, gamma_samples, param_samples, X):
    """
    code taken from https://github.com/probml/pyprobml/blob/master/notebooks/book2/19/bnn_mnist_sgld_blackjax.ipynb
   net_model: neural net model for predictions
   params_sample: params of neural net sampled using MCMC
   X: data

   how to calculate posterior preds?
   .......
   Let n_samples_of_params = 5
   For given Xi:
       model_predicted_labels = [2, 2, 2, 8, 0]
       then,
       pred_prob (x1) = freq of most repeated labels / n_samles = 3/5

   returns
   .......
   posterior predictive probabilities of size len(X)
   """

    NUM_CLS = 2

    def calc_prob(i, pred_cls):
        cls_cnt = jnp.bincount(pred_cls, length=NUM_CLS)
        total = jnp.sum(cls_cnt)
        prob_c_0 = cls_cnt[0] / total
        prob_c_1 = cls_cnt[1] / total

        return i+1, jnp.array([prob_c_0, prob_c_1])


    # predictive probabilities of X for each param
    predicted_class = jax.vmap(model_prediction, in_axes=(None, 0, None, 0))(model, param_samples, X, gamma_samples).squeeze()

    predicted_class = predicted_class.astype(jnp.int32).T
    # posterior predictive probability using histogram
    _, posterior_pred_probs = jax.lax.scan(calc_prob, 0, predicted_class)

    return posterior_pred_probs


def get_accuracy_vs_percentage_certainity(X, y, posterior_pred_probs, model, params_samples, gamma_samples):
    thresholds = jnp.arange(0, 1.1, 0.1)
    pcts = []
    accs = []
    for thr in thresholds:
        certain_mask = posterior_pred_probs >= thr

        # accuracy
        if certain_mask.sum() == 0:
            acc_sample = 1

        else:
            acc_sample = jax.vmap(compute_model_accuracy, in_axes=(0, 0, None, None, None))(
                params_samples, gamma_samples, model, X[certain_mask], y[certain_mask]
            ).mean()

        accs.append(acc_sample)

        # percentage of certainty
        pct = jnp.mean(certain_mask.mean())
        print(pct, acc_sample)
        pcts.append(pct)

    return accs, pcts

def plot_accuracy_perc_certainity(ax, accs, pcts, thresholds, bbox=(0.8, 0.8), show_legend=True):
    ax_l = ax

    # plot perc_certainity
    ax_r = ax_l.twinx()
    pct_plot = ax_r.plot(thresholds, pcts, "-+", color="green", label="pct of certain preds")
    ax_r.set_ylabel("pct")
    ax_l.set_xlabel("certainty threshold")
    ax_r.set_ylim(0, 1.05)

    # plot accuracy
    acc_plot = ax_l.plot(thresholds, accs, "-+", label="Certainty Accuracy")
    ax_l.set_ylabel("accuracy")

    # plot accuracy on whole batch
    mn, mx = ax_r.get_xlim()
    acc_hline = ax_l.hlines(accs[0], mn, mx, color="black", linestyle="-.", label="Test accuracy")
    if show_legend:
        ax_r.legend(handles=[acc_plot[0], pct_plot[0], acc_hline], bbox_to_anchor=bbox, frameon=False)

    return ax_l, ax_r

def batch_data(rng_key: PRNGKey, data: Batch, batch_size: int) -> Batch:
    """Return an iterator over batches of data."""
    data_size = data.x.shape[0]
    while True:
        _, key = jax.random.split(rng_key)
        idx = jax.random.choice(key=key, a=jnp.arange(data_size), shape=(batch_size,), replace=False)

        minibatch = Batch(data.x[idx], data.y[idx])
        yield minibatch

def get_ess(chain, burn_in):
    c = chain
    l = c.shape[0]
    bi = int(burn_in * l)
    c = c[bi:]
    cv = tfp.mcmc.effective_sample_size(c)
    cv = cv.at[jnp.isnan(cv)*1].set(1.)
    return cv

def running_average(old_avg_val, new_val, n_avg):
    new_avg_val = old_avg_val + (new_val - old_avg_val) / (n_avg + 1)
    return new_avg_val


def compute_updated_ensemble_predictions_classification(
        ensemble_predicted_probs, num_ensembled, new_predicted_probs):
    """Update ensemble predictive categorical distribution."""
    #ToDo: test
    if num_ensembled:
        new_ensemble_predicted_probs = running_average(ensemble_predicted_probs,
                                                       new_predicted_probs,
                                                       num_ensembled)
    else:
        new_ensemble_predicted_probs = new_predicted_probs
    return new_ensemble_predicted_probs

def make_constant_lr_schedule_with_cosine_burnin(init_lr, final_lr,
                                                 burnin_steps):
    """Cosine LR schedule with burn-in for SG-MCMC."""

    def schedule(step):
        t = jnp.minimum(step / burnin_steps, 1.)
        coef = (1 + jnp.cos(t * np.pi)) * 0.5
        return coef * init_lr + (1 - coef) * final_lr

    return schedule

def make_cyclical_cosine_lr_schedule(init_lr, total_steps, cycle_length):
    """Cosine LR schedule with burn-in for SG-MCMC."""

    def schedule(step):
        k = total_steps // cycle_length
        t = (step % k) / k
        coef = (1 + jnp.cos(t * np.pi)) * 0.5
        return coef * init_lr

    return schedule

def make_cyclical_cosine_lr_schedule_with_const_burnin(init_lr, burnin_steps,
                                                       cycle_length):

    def schedule(step):
        t = jnp.maximum(step - burnin_steps - 1, 0.)
        t = (t % cycle_length) / cycle_length
        return 0.5 * init_lr * (1 + jnp.cos(t * jnp.pi))

    return schedule

def make_step_size_fn(init_lr, schedule, alpha, n_samples,
                      cycle_len):
    if schedule == "constant":
        return lambda _: init_lr

    if schedule == "exponential":
        return optax.exponential_decay(init_lr, decay_rate=alpha,
                                       transition_begin=0, transition_steps=n_samples)

    if schedule == "cyclical":
        if cycle_len is None or cycle_len < 0:
            cycle_len = 10

        return make_cyclical_cosine_lr_schedule(init_lr, n_samples, cycle_len)

def make_batch(idx, x, y):
    return Batch(x[idx], y[idx])

def get_mixed_model_auc(model, params, data, gamma):
    logits = model.apply(params, data.x, gamma).ravel()
    pred_probs = jax.nn.sigmoid(logits)
    return roc_auc_score(data.y, pred_probs)

def get_model_auc(model, params, data):
    logits = model.apply(params, data.x).ravel()
    pred_probs = jax.nn.sigmoid(logits)
    return roc_auc_score(data.y, pred_probs)


def get_mixed_model_pred(model, params, x, gamma):
    logits = model.apply(params, x, gamma).ravel()
    pred_probs = jax.nn.sigmoid(logits)
    return pred_probs

def get_model_pred(model, params, x):
    logits = model.apply(params, x).ravel()
    pred_probs = jax.nn.sigmoid(logits)
    return pred_probs

def cross_entropy_loss(model, x, y, params, gamma):
    logits = model.apply(params, x, gamma).ravel()
    loss = optax.sigmoid_binary_cross_entropy(logits, y)
    return loss

def mse_loss(model, x, y, params, gamma):
    preds = model.apply(params, x, gamma).ravel()
    loss = jnp.mean(optax.l2_loss(preds, y))
    return loss

def fisher_exact_test(X, y, thres=0.05):
    cols = X.columns
    p_values = np.zeros(len(cols))
    for i, col in enumerate(cols):
        table = pd.crosstab(y, X[col])
        _, p_val = stats.fisher_exact(table, alternative="two-sided")
        p_values[i] = p_val

    idx_sig = np.argwhere(p_values < thres)
    print(f"Total of {len(idx_sig)} variables are significant (p_val = {thres})")

    return idx_sig


def build_network(X, net_intr, net_intr_rev):
    p = X.shape[1]
    J = np.zeros((p, p))
    cols = X.columns
    intrs = []
    intrs_rev = []
    for i, g1 in enumerate(cols):
        try:
            g_intrs = net_intr.loc[g1]
            if isinstance(g_intrs, int):
                g_intrs = [g_intrs]
            else:
                g_intrs = list(g_intrs)
            for g2 in g_intrs:
                if (g2, g1) not in intrs_rev: # check if we haven't encountered the reverse interaction
                    j = cols.get_loc(g2)
                    J[i, j] = 1.0
                    J[j, i] = 1.0
                    intrs.append((g1, g2))
        except KeyError:
            continue

        # Check the reverse direction
        try:
            g_intrs_rev = net_intr_rev.loc[g1]
            if isinstance(g_intrs_rev, int):
                g_intrs_rev = [g_intrs_rev]
            else:
                g_intrs_rev = list(g_intrs_rev)
            for g2 in g_intrs_rev:
                if (g1, g2) not in intrs:
                    j = cols.get_loc(g2)
                    J[i, j] = 1.0
                    J[j, i] = 1.0
                    intrs_rev.append((g2, g1))

        except KeyError:
            continue


    return J

def build_network_string(gene_names, string_ppi):

    net_intr = pd.Series(string_ppi["symbolA"].values, index=string_ppi["symbolB"])
    net_intr_rev = pd.Series(string_ppi["symbolB"].values, index=string_ppi["symbolA"]) 

    p = len(gene_names)
    J = np.zeros((p, p))
    intrs = []
    intrs_rev = []
    for i, g1 in enumerate(gene_names):
        try:
            g_intrs = net_intr.loc[g1]
            if isinstance(g_intrs, int):
                g_intrs = [g_intrs]
            else:
                g_intrs = list(g_intrs)
            for g2 in g_intrs:
                if (g2, g1) not in intrs_rev: # check if we haven't encountered the reverse interaction
                    if g2 in gene_names:
                        j = gene_names.index(g2)
                        weight = string_ppi[(string_ppi["symbolA"] == g1) & (string_ppi["symbolB"] == g2)]["weight"].values[0]
                        J[i, j] = weight
                        J[j, i] = weight
                        intrs.append((g1, g2))
        except KeyError:
            continue

        # Check the reverse direction
        try:
            g_intrs_rev = net_intr_rev.loc[g1]
            if isinstance(g_intrs_rev, int):
                g_intrs_rev = [g_intrs_rev]
            else:
                g_intrs_rev = list(g_intrs_rev)
            for g2 in g_intrs_rev:
                if (g1, g2) not in intrs:
                    if g2 in gene_names:
                        j = gene_names.index(g2)
                        weight = string_ppi[(string_ppi["symbolB"] == g1) & (string_ppi["symbolA"] == g2)]["weight"].values[0]
                        J[i, j] = weight
                        J[j, i] = weight
                        intrs_rev.append((g2, g1))

        except KeyError:
            continue


    return J

def setup_logger(log_path, seed):
    logging.getLogger().handlers = []
    logging.getLogger().setLevel(logging.NOTSET)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s], %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    if log_path is None or log_path == "":
        if not os.path.exists(f"{log_path}/logs"):
            os.makedirs(f"{log_path}/logs")

        rotatingHandler = logging.handlers.RotatingFileHandler(filename=f"{log_path}/logs/log_s_{seed}.log", maxBytes=(1048576*5),
                                                               backupCount=7)
        rotatingHandler.setLevel(logging.INFO)
        rotatingHandler.setFormatter(formatter)
        logging.getLogger().addHandler(rotatingHandler)
    log = logging.getLogger()
    return log

def integrated_gradients(model, params, gammas, x, N):

    p = x.shape[-1]
    baseline = jnp.zeros((1, p))
    t = jnp.linspace(0, 1, N).reshape(-1, 1)
    path = baseline * (1 - t) + x * t

    def get_grad(pi):
        # compute gradient
        # add/remove batch axes
        return jnp.mean(jax.vmap(jax.grad(lambda param, gamma, p: model.apply(param, gamma, p, True).squeeze(), argnums=2),
                                 in_axes=(0, 0, None))(params, gammas, pi), axis=0)

    gs = jax.vmap(get_grad)(path)
    # sum pieces (Riemann sum), multiply by (x - x')
    ig = jnp.mean(gs, axis=0) * (x.reshape(1, -1) - baseline)
    return ig

def evaluate_bnn_bg_models(model, X, y, params, gammas, loss_fn):
    eval_fn = lambda p, g: model.apply(p, g, X, True).ravel()
    # print(gammas.shape)
    # print(params["linear"]["w"].shape)
    preds = eval_fn(params, gammas)
    # print(preds.shape)
    # preds = preds.reshape(-1, preds.shape[-1])
    # losses = jax.vmap(lambda x, z: jnp.sqrt(jnp.mean((x - z)**2)), in_axes=(0, None))(preds, y)
    # mean_loss = jnp.sqrt(jnp.mean(losses, axis=-1))
    # return jnp.mean(losses)
    return loss_fn(y, preds)

def get_feats_dropout_loss(model, params, gammas, X, y, classifier=False):
    var_loss_dict = {"feats_idx": [], "num_models": [] , "loss_on": [], "loss_off": [], "loss_diff": []}

    p = X.shape[1]

    if classifier:
        loss_fn = lambda y, logits: optax.sigmoid_binary_cross_entropy(logits, y)

    else:
        loss_fn = lambda y, preds: jnp.sqrt(jnp.mean((y - preds)**2))

    for idx in tqdm(range(p)):
        idx_on = np.argwhere(gammas[:,idx] == 1.).ravel()
        loss_on, loss_off = 0., 0.
        if idx_on.size == 0: ## irrelevant feature
            loss_diff = 1e9
        else:
            disc_states_on = gammas[idx_on]
            # print(disc_states_on.shape)
            params_on = jax.tree_util.tree_map(lambda x: x[idx_on], params)
            loss_on = jnp.sum(jax.vmap(lambda p, g: evaluate_bnn_bg_models(model, X, y, p, g, loss_fn))(params_on, disc_states_on))
            # Turn-off the variable, and see how the loss changes
            disc_states_off = disc_states_on.at[:,idx].set(0)
            loss_off = jnp.sum(jax.vmap(lambda p, g: evaluate_bnn_bg_models(model, X, y, p, g, loss_fn))(params_on, disc_states_off))

            # loss_diff = (loss_on - loss_off) * (len(idx_on) / num_models)
            loss_diff = (loss_on - loss_off)


        var_loss_dict["feats_idx"].append(idx)
        var_loss_dict["num_models"].append(idx_on.size)
        var_loss_dict["loss_on"].append(loss_on)
        var_loss_dict["loss_off"].append(loss_off)
        var_loss_dict["loss_diff"].append(loss_diff)


    var_loss_df = pd.DataFrame(var_loss_dict).sort_values(by="loss_diff")

    return var_loss_df


def find_feats_on_graph(feat_idx, J):
    G = np.zeros((len(feat_idx), len(feat_idx)))
    for i, f1 in enumerate(feat_idx):
        for j, f2 in enumerate(feat_idx):
            if f1 != f2:
                G[i, j] = J[f1, f2]

    return G
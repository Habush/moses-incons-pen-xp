__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

# import numpy as np
from util import *
import autograd.numpy as np
from autograd import grad
import pandas as pd
import scipy

def p(z):
    return 1/(1 + np.exp(-z))

def grad_norm_2(b):
    if np.count_nonzero(b) == 0: return 0
    # return b / (np.sqrt(np.dot(b, b)))
    return 2*b
def grad_norm_1(b):
    if np.count_nonzero(b) == 0: return 0
    return b / (np.abs(b))

def quad_form(b, X, A):
    f = sigmoid(X @ b)
    return f.T @ A @ f

def grad_quad_form(b, X, A):
    f = sigmoid(X @ b)
    return 2 * (f @ f) * (A @ (1 - f) @ X)

## use the quadratic form of the laplacian with the coefficients
def quad_form_coef(b, A):
    return b.T @ A @ b

def grad_quad_form_coef(b, A):
    return 2 * A @ b

def objective_log_loss_l2(b, X, y, L, l1, l2):
    f = X @ b
    m = X.shape[0]
    log_ll = np.sum((y * f) - np.log(1 + np.exp(f))) / m
    return -log_ll + l1 * np.power(np.linalg.norm(b, 2), 2) \
                + l2 * quad_form(b, X, L)


def grad_log_loss_l2(b, X, y, L, l1, l2):
    h = sigmoid(X @ b)
    m = X.shape[0]
    grad_log = X.T @ (h - y)
    # print(f"grad: {grad_log}, m: {m}")
    return grad_log/m + l1 * grad_norm_2(b) + l2 * grad_quad_form(b, X, L)

def objective_log_loss_l2_coef(b, X, y, L, l1, l2):
    f = X @ b
    m = X.shape[0]
    log_ll = np.sum((y * f) - np.log(1 + np.exp(f))) / m
    return -log_ll + l1 * np.linalg.norm(b, 2) \
                + l2 * quad_form_coef(b, L)


def grad_log_loss_l2_coef(b, X, y, L, l1, l2):
    h = sigmoid(X.dot(b))
    m = X.shape[0]
    grad_log = X.T @ (h - y)
    # print(f"grad: {grad_log}, m: {m}")
    return grad_log/m + l1 * grad_norm_2(b) + l2 * grad_quad_form_coef(b, L)

def objective_log_loss_l1(b, X, y, L, l1, l2):
    f = X @ b
    m = X.shape[0]
    log_ll = np.sum((y*f) - np.log(1 + np.exp(f))) / m
    return -log_ll + l1 * np.sum(np.abs(b)) \
                + l2 * quad_form(b, X, L)


def grad_log_loss_l1(b, X, y, L, l1, l2):
    h = sigmoid(X @ b)
    m = X.shape[0]
    grad_log = X.T @ (h - y)
    return grad_log/m + l1 * grad_norm_1(b) + l2 * grad_quad_form(b, X, L)

def objective_log_loss_l1_coef(b, X, y, L, l1, l2):
    f = X @ b
    m = X.shape[0]
    log_ll = np.sum((y*f) - np.log(1 + np.exp(f))) / m
    return -log_ll + l1 * np.sum(np.abs(b)) \
                + l2 * quad_form_coef(b, L)


def grad_log_loss_l1_coef(b, X, y, L, l1, l2):
    h = sigmoid(X @ b)
    m = X.shape[0]
    grad_log = X.T @ (h - y)
    return grad_log/m + l1 * grad_norm_1(b) + l2 * grad_quad_form_coef(b, L)

def objective_log_loss_l2_no_pen(b, X, y, l1):
    f = X @ b
    m = X.shape[0]
    log_ll = np.sum((y * f) - np.log(1 + np.exp(f))) / m
    return -log_ll + l1 * np.linalg.norm(b, 2)


def grad_log_loss_l2_no_pen(b, X, y, l1):
    h = sigmoid(X @ b)
    m = X.shape[0]
    grad_log = X.T @ (h - y)
    # print(f"grad: {grad_log}, m: {m}")
    return grad_log/m + l1 * grad_norm_2(b)

def objective_log_loss_l1_no_pen(b, X, y, l1):
    f = X @ b
    m = X.shape[0]
    log_ll = np.sum((y*f) - np.log(1 + np.exp(f))) / m
    return -log_ll + l1 * np.sum(np.abs(b))


def grad_log_loss_l1_no_pen(b, X, y, l1):
    h = sigmoid(X @ b)
    m = X.shape[0]
    grad_log = X.T @ (h - y)
    return grad_log/m + l1 * grad_norm_1(b)


def solve_logistic_reg_grad(X_train, X_test, y_train, y_test, l1_vals, l2_vals=None, gamma=None, assoc_mat=None,
                                err_fn=log_loss_cp, l2_norm=True, lap_norm=False, use_coef=False):


    beta_0 = np.random.rand(X_train.shape[1])

    if l2_vals is not None:
        train_errors = np.full((l1_vals.shape[0], l2_vals.shape[0]), 1e6)  # set default to 1e6 to filter out cases where the solver failed
        test_errors = np.full((l1_vals.shape[0], l2_vals.shape[0]), 1e6)
        beta_vals = np.zeros((l1_vals.shape[0], l2_vals.shape[0], X_train.shape[1]))
        ll_pens = np.full((l1_vals.shape[0], l2_vals.shape[0]), 1e6)
        l1_pens = np.full((l1_vals.shape[0], l2_vals.shape[0]), 1e6)
        l2_pens = np.full((l1_vals.shape[0], l2_vals.shape[0]), 1e6)

        if use_coef:
            L = scipy.sparse.csgraph.laplacian(assoc_mat, normed=lap_norm)
        else:
            prec_mat = get_emp_covariance(X_train, assoc_mat)

            L = get_laplacian_mat(X_train, X_train, prec_mat, gamma, lap_norm)

        for i, l1 in enumerate(l1_vals):
            for j, l2 in enumerate(l2_vals):
                # print(f"[{datetime.now()}] - Using l1 - {l1}, l2 - {l2}")
                if use_coef:
                    if l2_norm:
                        beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l2_coef, x0=beta_0, fprime=grad(objective_log_loss_l2_coef, 0), args=(X_train, y_train, L, l1, l2), maxiter=1000)[0]
                        # beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l2_coef, x0=beta_0,  approx_grad=True, args=(X_train, y_train, L, l1, l2), maxiter=1000)[0]
                    else:
                        beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l1_coef, x0=beta_0, fprime=grad(objective_log_loss_l1_coef, 0),
                                                        args=(X_train, y_train, L, l1, l2), maxiter=1000)[0]
                        # beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l1_coef, x0=beta_0,
                        #                                     approx_grad=True,
                        #                                     args=(X_train, y_train, L, l1, l2), maxiter=1000)[0]

                else:
                    if l2_norm:
                        beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l2, x0=beta_0, fprime=grad(objective_log_loss_l2, 0), args=(X_train, y_train, L, l1, l2), maxiter=1000)[0]
                        # beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l2, x0=beta_0, approx_grad=True,
                        #                                     args=(X_train, y_train, L, l1, l2), maxiter=1000)[0]
                    else:
                        beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l1, x0=beta_0, fprime=grad(objective_log_loss_l1, 0),
                                                            args=(X_train, y_train, L, l1, l2), maxiter=1000)[0]
                        # beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l1, x0=beta_0, approx_grad=True,
                        #                                     args=(X_train, y_train, L, l1, l2), maxiter=1000)[0]

                train_errors[i, j] = err_fn(X_train, y_train, beta)
                test_errors[i, j] = err_fn(X_test, y_test, beta)
                beta_vals[i, j] = beta

                log_l, c1, c2 = get_penalty_comp_log(X_train, y_train, beta, L, use_coef=use_coef)
                ll_pens[i, j], l1_pens[i, j], l2_pens[i, j] = log_l, c1, c2

        return train_errors, test_errors, beta_vals, ll_pens, l1_pens, l2_pens
    else:
        train_errors = np.full(l1_vals.shape[0], 1e6)  # set default to 1e6 to filter out cases where the solver failed
        test_errors = np.full(l1_vals.shape[0], 1e6)
        beta_vals = np.zeros((l1_vals.shape[0], X_train.shape[1]))
        ll_pens = np.full(l1_vals.shape[0], 1e6)
        l1_pens = np.full(l1_vals.shape[0], 1e6)

        for i, l1 in enumerate(l1_vals):
            if l2_norm:
                beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l2_no_pen, x0=beta_0, fprime=grad(objective_log_loss_l2_no_pen, 0),
                                                    args=(X_train, y_train, l1), maxiter=1000)[0]
                # beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l2_no_pen, x0=beta_0, approx_grad=True,
                #                              args=(X_train, y_train, l1), maxiter=1000)[0]
            else:
                beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l1_no_pen, x0=beta_0, fprime=grad(objective_log_loss_l1_no_pen, 0),
                                                    args=(X_train, y_train, l1), maxiter=1000)[0]
                # beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l1_no_pen, x0=beta_0, approx_grad=True,
                #                              args=(X_train, y_train, l1), maxiter=1000)[0]
            train_errors[i] = err_fn(X_train, y_train, beta)
            test_errors[i] = err_fn(X_test, y_test, beta)
            beta_vals[i] = beta

            log_l, c1 = get_penalty_comp_log(X_train, y_train, beta)
            ll_pens[i], l1_pens[i]= log_l, c1,

        return train_errors, test_errors, beta_vals, ll_pens, l1_pens

def apply_logisitc_reg_grad(X_train, X_test, y_train, y_test, l1, l2=None, gamma=None, assoc_mat=None,
                                err_fn=log_loss_cp, l2_norm=True, lap_norm=False, use_coef=False):

    beta_0 = np.random.rand(X_train.shape[1])

    if l2 is not None:

        if use_coef:
            L = scipy.sparse.csgraph.laplacian(assoc_mat, normed=lap_norm)
            if l2_norm:
                beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l2_coef, x0=beta_0, fprime=grad(objective_log_loss_l2_coef, 0),
                                                    args=(X_train, y_train, L, l1, l2), maxiter=1000)[0]
            else:
                beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l1_coef, x0=beta_0, fprime=grad(objective_log_loss_l1_coef, 0),
                                                    args=(X_train, y_train, L, l1, l2), maxiter=1000)[0]
        else:
            prec_mat = get_emp_covariance(X_train, assoc_mat)

            L = get_laplacian_mat(X_train, X_train, prec_mat, gamma, lap_norm)

            if l2_norm:
                beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l2, x0=beta_0, fprime=grad(objective_log_loss_l2, 0), args=(X_train, y_train, L, l1, l2), maxiter=1000)[0]
                # beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l2, x0=beta_0, approx_grad=True,
                #                                     args=(X_train, y_train, L, l1, l2), maxiter=1000)[0]
            else:
                beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l1, x0=beta_0, fprime=grad(objective_log_loss_l1, 0),
                                                    args=(X_train, y_train, L, l1, l2), maxiter=1000)[0]
                # beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l1, x0=beta_0, approx_grad=True,
                #                                     args=(X_train, y_train, L, l1, l2), maxiter=1000)[0]
        train_error = err_fn(X_train, y_train, beta)
        test_error = err_fn(X_test, y_test, beta)
        beta_vals = np.ndarray.flatten(beta)

        log_l, c1, c2 = get_penalty_comp_log(X_train, y_train, beta, L, use_coef=use_coef)

        return train_error, test_error, beta_vals, log_l, c1, c2

    else:
        if l2_norm:
            beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l2_no_pen, x0=beta_0, fprime=grad(objective_log_loss_l2_no_pen, 0),
                                                args=(X_train, y_train, l1), maxiter=1000)[0]
            # beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l2_no_pen, x0=beta_0, approx_grad=True,
            #                                     args=(X_train, y_train, l1), maxiter=1000)[0]
        else:
            beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l1_no_pen, x0=beta_0, fprime=grad(objective_log_loss_l1_no_pen, 0),
                                                args=(X_train, y_train, l1), maxiter=1000)[0]
            # beta = scipy.optimize.fmin_l_bfgs_b(objective_log_loss_l1_no_pen, x0=beta_0, approx_grad=True,
            #                                     args=(X_train, y_train, l1), maxiter=1000)[0]
        train_error = err_fn(X_train, y_train, beta)
        test_error = err_fn(X_test, y_test, beta)
        beta_vals = np.ndarray.flatten(beta)

        log_l, c1 = get_penalty_comp_log(X_train, y_train, beta)

        return train_error, test_error, beta_vals, log_l, c1

def run_logisitic_reg_exp(X_train, X_test, y_train, y_test, assoc_mat, gammas, l1_vals, l2_vals,
                                l2_norm=True, lap_norm=False, use_coef=False):

    # train_errs_1 = np.zeros(len(gammas))
    train_errs_cv = np.zeros(len(gammas))
    test_errs = np.zeros(len(gammas))
    # beta_vals_cv_2 = np.zeros((len(gammas), n_iter, n_iter, X_train.shape[1]))
    beta_vals = np.zeros((len(gammas), X_train.shape[1]))

    train_ll_pen, test_ll_pen = np.zeros(len(gammas)), np.zeros(len(gammas))
    train_l1_pen, test_l1_pen = np.zeros(len(gammas)), np.zeros(len(gammas))
    train_l2_pen, test_l2_pen = np.zeros(len(gammas)), np.zeros(len(gammas))
    min_l1_vals, min_l2_vals = np.zeros(len(gammas)), np.zeros(len(gammas))

    n_folds = 5
    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    for i, g in enumerate(gammas):

        fold_test_errs = np.zeros((n_folds, l1_vals.shape[0], l2_vals.shape[0]))
        fold_beta_vals = np.zeros((n_folds, l1_vals.shape[0], l2_vals.shape[0], X_train.shape[1]))
        fold_ll_pen = np.zeros((n_folds, l1_vals.shape[0], l2_vals.shape[0]))
        fold_l1_pen = np.zeros((n_folds, l1_vals.shape[0], l2_vals.shape[0]))
        fold_l2_pen = np.zeros((n_folds, l1_vals.shape[0], l2_vals.shape[0]))
        j = 0
        print(f"[{datetime.now()}] - gamma - {g:.2f}")
        for train_idx, test_idx in cv.split(X_train, y_train):
            # print(f"[{datetime.now()}] - fold - {j + 1}")
            x_train_cv, x_test_cv = X_train[train_idx], X_train[test_idx]
            y_train_cv, y_test_cv = y_train[train_idx], y_train[test_idx]
            _,  fold_test_errs[j], fold_beta_vals[j],\
                    fold_ll_pen[j], fold_l1_pen[j], fold_l2_pen[j] = solve_logistic_reg_grad(x_train_cv, x_test_cv, y_train_cv, y_test_cv, l1_vals,
                                                                             l2_vals=l2_vals, gamma=g, assoc_mat=assoc_mat, l2_norm=l2_norm, use_coef=use_coef)

            j += 1
        fold_cv_err = np.mean(fold_test_errs, axis=0)
        min_idx = np.unravel_index(np.argmin(fold_cv_err), fold_cv_err.shape)

        train_errs_cv[i] = fold_cv_err[min_idx]
        train_ll_pen[i] = np.mean(fold_ll_pen, axis=0)[min_idx]
        train_l1_pen[i] = np.mean(fold_l1_pen, axis=0)[min_idx]

        train_l2_pen[i] = np.mean(fold_l2_pen, axis=0)[min_idx]

        min_l1_val, min_l2_val = l1_vals[min_idx[0]], l2_vals[min_idx[1]]
        min_l1_vals[i] = min_l1_val
        min_l2_vals[i] = min_l2_val
        _, test_errs[i], beta_vals[i], \
        test_ll_pen[i], test_l1_pen[i], test_l2_pen[i] = apply_logisitc_reg_grad(X_train, X_test, y_train, y_test,
                                                                                 min_l1_val, l2=min_l2_val, gamma=g,
                                                                                 assoc_mat=assoc_mat, l2_norm=l2_norm, lap_norm=lap_norm, use_coef=use_coef)

        print(f"Min l1: {min_l1_val}, Min l2: {min_l2_val}, CV Error: {fold_cv_err[min_idx]}, Test error: {test_errs[i]}\n"
            f"ll_pen_train: {train_ll_pen[i]}, l1_pen_train: {train_l1_pen[i]}, l2_pen_train: {train_l2_pen[i]}")



    return train_errs_cv, test_errs, train_ll_pen, test_ll_pen, train_l1_pen, test_l1_pen, train_l2_pen, test_l2_pen, min_l1_vals, min_l2_vals, beta_vals



def run_logisitic_reg_exp_no_bp(X_train, X_test, y_train, y_test, l1_vals, l2_norm=False):

    n_folds = 5
    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    fold_test_errs = np.zeros((n_folds, l1_vals.shape[0]))
    fold_beta_vals = np.zeros((n_folds, l1_vals.shape[0], X_train.shape[1]))
    fold_ll_pen = np.zeros((n_folds, l1_vals.shape[0]))
    fold_l1_pen = np.zeros((n_folds, l1_vals.shape[0]))

    j = 0
    for train_idx, test_idx in cv.split(X_train, y_train):
        # print(f"[{datetime.now()}] - fold - {j + 1}")
        x_train_cv, x_test_cv = X_train[train_idx], X_train[test_idx]
        y_train_cv, y_test_cv = y_train[train_idx], y_train[test_idx]
        _, fold_test_errs[j], fold_beta_vals[j], \
        fold_ll_pen[j], fold_l1_pen[j] = solve_logistic_reg_grad(x_train_cv, x_test_cv, y_train_cv, y_test_cv, l1_vals, l2_norm=l2_norm)
        j += 1
    fold_cv_err = np.mean(fold_test_errs, axis=0)
    min_idx = np.unravel_index(np.argmin(fold_cv_err), fold_cv_err.shape)

    train_errs_cv = fold_cv_err[min_idx]
    train_ll_pen = np.mean(fold_ll_pen, axis=0)[min_idx]
    train_l1_pen = np.mean(fold_l1_pen, axis=0)[min_idx]
    min_l1_val = l1_vals[min_idx[0]]
    min_l1_vals = min_l1_val

    _, test_errs, beta_vals, \
    test_ll_pen, test_l1_pen = apply_logisitc_reg_grad(X_train, X_test, y_train, y_test, min_l1_val, l2_norm=l2_norm)

    print(f"Min l1: {min_l1_val}, CV Error: {fold_cv_err[min_idx]}, Test error: {test_errs}\n"
          f"ll_pen_train: {train_ll_pen}, l1_pen_train: {train_l1_pen}")

    return train_errs_cv, test_errs, train_ll_pen, test_ll_pen, train_l1_pen, test_l1_pen, min_l1_vals, beta_vals

def build_table_from_log(gammas, train_errs_cv, test_errs, train_ll_pen, test_ll_pen, train_l1_pen, test_l1_pen, min_l1_vals, train_l2_pen=None, test_l2_pen=None, min_l2_vals=None, lap=True):

    if lap:
        res = {"gamma": [], "min_train_err_cv": [], "l1_train_cv": [], "l2_train_cv": [],
                 "ll_pen_train_cv": [], "l1_pen_train_cv": [], "l2_pen_train_cv": [],
                 "test_err": [], "ll_pen_test": [], "l1_pen_test": [], "l2_pen_test": []
                 }

        for i, g in enumerate(gammas):
            res["gamma"].append(g)
            res["min_train_err_cv"].append(train_errs_cv[i])
            res["l1_train_cv"].append(min_l1_vals[i])
            res["l2_train_cv"].append(min_l2_vals[i])
            res["ll_pen_train_cv"].append(train_ll_pen[i])
            res["l1_pen_train_cv"].append(train_l1_pen[i])
            res["l2_pen_train_cv"].append(train_l2_pen[i])

            res["test_err"].append(test_errs[i])
            res["ll_pen_test"].append(test_ll_pen[i])
            res["l1_pen_test"].append(test_l1_pen[i])
            res["l2_pen_test"].append(test_l2_pen[i])

        res_df = pd.DataFrame(res)
        return res_df
    else:
        res = {"min_train_err_cv": [], "l1_train_cv": [],
               "ll_pen_train_cv": [], "l1_pen_train_cv": [],
               "test_err": [], "ll_pen_test": [], "l1_pen_test": []}

        res["min_train_err_cv"].append(train_errs_cv)
        res["l1_train_cv"].append(min_l1_vals)
        res["ll_pen_train_cv"].append(train_ll_pen)
        res["l1_pen_train_cv"].append(train_l1_pen)

        res["test_err"].append(test_errs)
        res["ll_pen_test"].append(test_ll_pen)
        res["l1_pen_test"].append(test_l1_pen)

        res_df = pd.DataFrame(res)
        return res_df
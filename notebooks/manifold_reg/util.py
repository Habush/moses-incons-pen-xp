__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

import math
import os.path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.pyplot import figure
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, log_loss, confusion_matrix
from sklearn.metrics import precision_score
from sklearn.model_selection import KFold, StratifiedKFold
import dask
from datetime import datetime

def is_positive_semidefinite(X):
    if X.shape[0] != X.shape[1]: # must be a square matrix
        return False

    if not np.all( X - X.T == 0 ): # must be a symmetric
        return False

    try: # Cholesky decomposition fails for matrices that are NOT positive definite.

        # But since the matrix may be positive SEMI-definite due to rank deficiency
        # we must regularize.
        regularized_X = X + np.eye(X.shape[0]) * 1e-14

        np.linalg.cholesky(regularized_X)
    except np.linalg.LinAlgError:
        return False

    return True

def sigmoid(z):
  return 1/(1 + np.exp(-z))

def loss_fn(X, Y, beta):
    return cp.norm(X @ beta - Y) ** 2

def reg1(beta):
    return cp.norm1(beta)

def reg2(beta, L):
    return cp.atoms.quad_form(beta, L)

def objective_fn(X, Y, beta, Lap, l1, l2):
    return loss_fn(X, Y, beta) + l1 * reg1(beta) + l2 * reg2(beta, Lap)

def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value

def precision(X, Y, beta):
    y_pred = X @ beta.value
    prec = precision_score(Y, y_pred)
    return prec

def solve_lasso(X_train, X_test, y_train, y_test, l1_vals, l2_vals, L, fp, n_iter=10):
    beta = cp.Variable((X_train.shape[1], 1))
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    l1, l2 = cp.Parameter(nonneg=True),  cp.Parameter(nonneg=True)
    prob = cp.Problem(cp.Minimize(objective_fn(X_train, y_train, beta, L, l1, l2)))

    train_errors = np.full((n_iter, n_iter), 1e6) # set default to 1e6 to filter out cases where the solver failed
    test_errors = np.full((n_iter, n_iter), 1e6)
    beta_vals = np.zeros((n_iter,  n_iter, X_train.shape[1]))

    i = 0
    j = 0
    for v in l1_vals:
        l1.value = v
        for u in l2_vals:
            l2.value = u

            try:
                prob.solve()
            except cp.SolverError:
                print(f"Encountered solver error for l1 - {l1.value}, l2 - {l2.value}. Moving on...")
                fp.write(f"Encountered solver error for l1 - {l1.value}, l2 - {l2.value}. Moving on...")
                continue
            train_errors[i, j] = mse(X_train, y_train, beta)
            test_errors[i, j]= mse(X_test, y_test, beta)
            beta_vals[i, j] = np.ndarray.flatten(beta.value)
            j += 1

        j = 0
        i += 1

    train_idx = np.unravel_index(np.argmin(train_errors), train_errors.shape)
    test_idx = np.unravel_index(np.argmin(test_errors), test_errors.shape)
    print(test_idx)
    lst_train_err = train_errors[train_idx]
    lst_test_err = test_errors[test_idx]
    best_beta = beta_vals[test_idx]
    l1_val, l2_val = l1_vals[test_idx[0]], l2_vals[test_idx[1]]
    return lst_train_err, lst_test_err, best_beta, l1_val, l2_val

def objective_log_fn(X, Y, beta, L, l1, l2):
    f = X @ beta
    log_ll = cp.sum(cp.multiply(Y, f) - cp.logistic(f))
    m = X.shape[0]
    L_psd = cp.atoms.affine.wraps.psd_wrap(L) # this is to avoid the numerical issue that arises in ARPACK
                                      # when checking the matrix is PSD. See https://github.com/cvxpy/cvxpy/issues/1424

    obj = log_ll / m - l1 * cp.norm(beta, 2) - l2 * cp.atoms.quad_form(f, L_psd)

    return obj

def objective_log_fn_2(X, Y, beta, L, l1, l2=None):
    f = X @ beta
    log_ll = cp.sum(cp.multiply(Y, f) - cp.logistic(f))
    m = X.shape[0]
    L_psd = cp.atoms.affine.wraps.psd_wrap(L) # this is to avoid the numerical issue that arises in ARPACK
                                      # when checking the matrix is PSD. See https://github.com/cvxpy/cvxpy/issues/1424
    if l2 is None:
        obj = -(log_ll / m) + l1 *cp.norm(beta, 2) + cp.atoms.quad_form(beta, L_psd)
    else:
        obj = -(log_ll / m) + l1 * cp.norm(beta, 2) +  l2 * cp.atoms.quad_form(beta, L_psd)
    return obj

def get_penalty_comp_log(X, y, beta, L=None):
    f = X @ beta
    m = X.shape[0]
    log_ll = np.sum((y * f) - np.log(1 + np.exp(f))) / m
    c1 = np.linalg.norm(beta, 2)

    if L is None:
        return -log_ll, c1
    else:
        c2 = f.T @ L @ f

        return -log_ll, c1, c2

def log_loss_cp(X, y, beta):
    p = sigmoid(X @ beta)
    return log_loss(y, p)


def get_laplacian_mat(X1, X2, prec_mat, gamma):

    K = calculate_mahal_kernel(X1, X2, prec_mat, gamma=gamma)
    # W =  np.sum(K, axis=0)
    # D = np.diag(W)
    #
    # L = D - K
    L = scipy.sparse.csgraph.laplacian(K, normed=False)
    return L


def solve_prob(prob):
    prob.solve(solver="SCS", max_iters=10000)
    return prob.variables()[0].value



def solve_logistic_reg(X_train, X_test, y_train, y_test, l1_vals, l2_vals, gamma, assoc_mat, err_fn=log_loss_cp):

    assert l1_vals.shape[0] == l2_vals.shape[0]

    n_iter = l1_vals.shape[0]

    train_errors = np.full((n_iter, n_iter), 1e6)  # set default to 1e6 to filter out cases where the solver failed
    test_errors = np.full((n_iter, n_iter), 1e6)
    beta_vals = np.zeros((n_iter, n_iter, X_train.shape[1]))
    ll_pens = np.full((n_iter, n_iter), 1e6)
    l1_pens = np.full((n_iter, n_iter), 1e6)
    l2_pens = np.full((n_iter, n_iter), 1e6)

    prec_mat = get_emp_covariance(X_train, assoc_mat)

    L = get_laplacian_mat(X_train, X_train, prec_mat, gamma)

    prob_lst = []

    for l1 in l1_vals:
        for l2 in l2_vals:
            beta = cp.Variable(shape=X_train.shape[1])
            prob = cp.Problem(cp.Maximize(objective_log_fn(X_train, y_train, beta, L, l1, l2)))
            prob_lst.append(prob)

    dasklist = [dask.delayed(solve_prob)(prob) for prob in prob_lst]
    results = dask.compute(*dasklist, scheduler='processes')

    for i in range(n_iter * n_iter):
        idx = np.unravel_index(i, (n_iter, n_iter))
        beta = results[i]
        train_errors[idx] = err_fn(X_train, y_train, beta)
        test_errors[idx] = err_fn(X_test, y_test, beta)
        beta_vals[idx] = beta

        log_l, c1, c2 = get_penalty_comp_log(X_train, y_train, beta, L)
        ll_pens[idx], l1_pens[idx], l2_pens[idx] = log_l, c1, c2

    return train_errors, test_errors, beta_vals, ll_pens, l1_pens, l2_pens

def apply_logisitc_reg(X_train, X_test, y_train, y_test, l1, l2, gamma, assoc_mat, err_fn=log_loss_cp):
    beta = cp.Variable(X_train.shape[1])

    prec_mat = get_emp_covariance(X_train, assoc_mat)

    L = get_laplacian_mat(X_train, X_train, prec_mat, gamma)

    prob = cp.Problem(cp.Maximize(objective_log_fn(X_train, y_train, beta, L, l1, l2)))
    prob.solve(solver="SCS", max_iters=10000)

    train_error = err_fn(X_train, y_train, beta.value)
    test_error = err_fn(X_test, y_test, beta.value)
    beta_vals = np.ndarray.flatten(beta.value)

    log_l, c1, c2 = get_penalty_comp_log(X_train, y_train, beta.value, L)

    return train_error, test_error, beta_vals, log_l, c1, c2

def generate_betas(sz, tf, n_genes, val_tf, val_genes, num_pos=-1):
    if num_pos > n_genes:
        raise ValueError(f"Number of positive genes {num_pos} must be less than or equal to number of genes {n_genes}. To set all "
                         f"genes to positive, set it to -1")
    beta = np.zeros(sz)

    k = (tf + (tf * n_genes))
    st = n_genes + 1
    m = 0
    for i in range(k):
        if i % st == 0:
            beta[i] = val_tf[m]
            z = 0
            for j in range(i + 1, i + st):
                if num_pos < 0:
                    beta[j] = val_tf[m]/val_genes
                else:
                    if z < num_pos:
                        beta[j] = val_tf[m]/val_genes
                    else:
                        beta[j] = -val_tf[m]/val_genes
                    z += 1
            m += 1

    return beta


def generate_data_v2(tf, genes, tf_on=4, corr=0.7, val_tf=None, val_gene=math.sqrt(10), num_pos=-1, n=100):
    if val_tf is None:
        val_tf = [2, 2, 2, 2]
    sz = (tf + (tf * genes))
    X = np.zeros((n, sz))
    assert len(val_tf) == tf_on
    m = genes + 1
    for t in range(0, m*tf, m):
        X_tf = np.random.normal(0, 1, size=n)
        X[:,t] = X_tf
        for g in range(t + 1, t + m):
            X_g = np.random.normal(corr*X_tf, 0.51, size=n)
            X[:,g] = X_g

    beta = generate_betas(sz, tf_on, genes, val_tf, val_gene, num_pos=num_pos)

    sd_e = math.sqrt(np.sum(np.square(beta)) / 4)
    err = np.random.normal(np.zeros(n), sd_e, size=n)

    y = (X @ beta) + err

    return X, beta, y


def generate_log_data(tf, genes, tf_on=4, corr=0.7, val_tf=None, val_gene=math.sqrt(10), num_pos=-1, n=100):
   if val_tf is None:
       val_tf = [2, 2, 2, 2]
   X, beta, y_lin = generate_data_v2(tf, genes, tf_on=tf_on, corr=corr, val_tf=val_tf, val_gene=val_gene, num_pos=num_pos, n=n)

   p = sigmoid(y_lin)
   y_log = scipy.stats.bernoulli.rvs(p, size=X.shape[0])

   return X, beta, y_log

def get_psd_mat(X):
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         if i != j:
    #             try:
    #                 X[i, j] = 0.5 * min(X[i, j], math.sqrt(X[i, i] * X[j, j]))
    #             except ValueError:
    #                 print(i, j)
    #                 raise ValueError("Math error")

    regularized_X = X + np.eye(X.shape[0]) * 1e-14
    return regularized_X

# @numba.jit(parallel=True, nopython=True)
def get_mahala_dist(X, Y, cov_inv):
    D = np.zeros((X.shape[0], Y.shape[0]))
    for n in range(X.shape[0]):
        for m in range(Y.shape[0]):
            diff = X[n] - X[m]
            D[n, m] = math.sqrt(diff @ cov_inv @ diff.T)
    return D

def get_emp_covariance(X, M):
    cov = np.cov(X, rowvar=False)
    ##sparsify the precision matrix
    # print(f"Cov rank: {np.linalg.matrix_rank(cov)}")
    # sprec = np.multiply(M, np.linalg.inv(cov))  #this one gives math erros

    sprec = np.multiply(M, scipy.linalg.pinv(cov, rtol=1e-5))

    sprec_psd = get_psd_mat(sprec)

    return sprec_psd

def calculate_mahal_kernel(X, Y, cov_inv, gamma=1.0):
    D = get_mahala_dist(X, Y, cov_inv)

    K = np.exp((-1.0/(2.0*(math.pow(gamma, 2)))) * np.square(D))

    return K

def binarize_with_median(X, med=None, upper=1, lower=0):
    X_bin = np.empty(X.shape)
    if med is None:
        med = np.median(X, axis=0)
    it = np.nditer([X, med], flags=["multi_index"])
    for x, y in it:
        X_bin[it.multi_index] = upper if x > y else lower

    return X_bin

def compare_kernels(gammas, X_train, X_test, y_train, y_test, assoc_mat, cv_fold=5, random_state=42):

    train_errs_mahal_lin = np.zeros(gammas.shape[0])
    train_errs_mahal_log = np.zeros(gammas.shape[0])
    train_errs_mahal_log_c = np.zeros(gammas.shape[0])
    train_errs_idt_lin = np.zeros(gammas.shape[0])
    train_errs_idt_log = np.zeros(gammas.shape[0])

    test_errs_mahal_lin = np.zeros(gammas.shape[0])
    test_errs_mahal_log = np.zeros(gammas.shape[0])
    test_errs_mahal_log_c = np.zeros(gammas.shape[0])
    test_errs_idt_lin = np.zeros(gammas.shape[0])
    test_errs_idt_log = np.zeros(gammas.shape[0])

    clf_mahal_lin = LinearRegression()
    clf_mahal_log = LogisticRegression()
    clf_mahal_log_c = LogisticRegression()

    clf_idt_lin = LinearRegression()
    clf_idt_log = LogisticRegression()

    k_cv = KFold(n_splits=cv_fold, random_state=random_state, shuffle=True)
    sk_cv = StratifiedKFold(n_splits=cv_fold, random_state=random_state, shuffle=True)

    X_train_bin = binarize_with_median(X_train)
    X_test_bin = binarize_with_median(X_test)
    y_train_bin = binarize_with_median(y_train)
    y_test_bin = binarize_with_median(y_test)



    for i, g in enumerate(gammas):
        val_scores = np.zeros((5, cv_fold))
        j = 0
        for train_idx, test_idx in k_cv.split(X_train):
            x_train_cv, x_train_bin_cv = X_train[train_idx], X_train_bin[train_idx]
            x_test_cv, x_test_bin_cv = X_train[test_idx], X_train_bin[test_idx]
            y_train_cv, y_train_bin_cv = y_train[train_idx], y_train_bin[train_idx]
            y_test_cv, y_test_bin_cv = y_train[test_idx], y_train_bin[test_idx]

            cov_mat_reg_cv = get_emp_covariance(x_train_cv, assoc_mat)
            cov_mat_bin_cv = get_emp_covariance(x_train_bin_cv, assoc_mat)

            idt_ker_cv = np.identity(x_train_cv.shape[1])
            K_train_mahal_lin_cv = calculate_mahal_kernel(x_train_cv, x_train_cv, cov_mat_reg_cv, gamma=g)
            K_test_mahal_lin_cv = calculate_mahal_kernel(x_test_cv, x_train_cv, cov_mat_reg_cv, gamma=g)

            K_train_mahal_log_cv = calculate_mahal_kernel(x_train_bin_cv, x_train_bin_cv, cov_mat_bin_cv, gamma=g)
            K_test_mahal_log_cv = calculate_mahal_kernel(x_test_bin_cv, x_train_bin_cv, cov_mat_bin_cv, gamma=g)


            K_train_idt_lin_cv = calculate_mahal_kernel(x_train_cv, x_train_cv, idt_ker_cv, gamma=g)
            K_test_idt_lin_cv = calculate_mahal_kernel(x_test_cv, x_train_cv, idt_ker_cv, gamma=g)

            K_train_idt_log_cv = calculate_mahal_kernel(x_train_bin_cv, x_train_bin_cv, idt_ker_cv, gamma=g)
            K_test_idt_log_cv = calculate_mahal_kernel(x_test_bin_cv, x_train_bin_cv, idt_ker_cv, gamma=g)

            clf_mahal_lin.fit(K_train_mahal_lin_cv, y_train_cv)
            clf_mahal_log.fit(K_train_mahal_log_cv, y_train_bin_cv)
            clf_mahal_log_c.fit(K_train_mahal_lin_cv, y_train_bin_cv)

            clf_idt_lin.fit(K_train_idt_lin_cv, y_train_cv)
            clf_idt_log.fit(K_train_idt_log_cv, y_train_bin_cv)


            val_scores[0][j] = mean_squared_error(y_test_cv, clf_mahal_lin.predict(K_test_mahal_lin_cv))
            val_scores[1][j] = log_loss(y_test_bin_cv, clf_mahal_log.predict_proba(K_test_mahal_log_cv))
            val_scores[2][j] = log_loss(y_test_bin_cv, clf_mahal_log_c.predict_proba(K_test_mahal_lin_cv))

            val_scores[3][j] = mean_squared_error(y_test_cv, clf_idt_lin.predict(K_test_idt_lin_cv))
            val_scores[4][j] = log_loss(y_test_bin_cv, clf_idt_log.predict_proba(K_test_idt_log_cv))

            j += 1

        # print(f"--- gamma val: {g} ---\nlin_mahala: {-np.mean(cv_scores_m_lin)}\nlog_mahala: {-np.mean(cv_scores_m_log)}"
        #       f"\nlin_idt:{-np.mean(cv_scores_i_lin)}\nlog_idt: {-np.mean(cv_scores_i_log)}\n")
        train_errs_mahal_lin[i] = np.mean(val_scores[0])
        train_errs_mahal_log[i] = np.mean(val_scores[1])
        train_errs_mahal_log_c[i] = np.mean(val_scores[2])
        train_errs_idt_lin[i] = np.mean(val_scores[3])
        train_errs_idt_log[i] = np.mean(val_scores[4])

        cov_mat_reg = get_emp_covariance(X_train, assoc_mat)
        cov_mat_bin = get_emp_covariance(X_train_bin, assoc_mat)

        idt_ker = np.identity(X_train.shape[1])
        K_train_mahal_lin = calculate_mahal_kernel(X_train, X_train, cov_mat_reg, gamma=g)
        K_test_mahal_lin = calculate_mahal_kernel(X_test, X_train, cov_mat_reg, gamma=g)

        K_train_mahal_log = calculate_mahal_kernel(X_train_bin, X_train_bin, cov_mat_bin, gamma=g)
        K_test_mahal_log = calculate_mahal_kernel(X_test_bin, X_train_bin, cov_mat_bin, gamma=g)

        K_train_idt_lin = calculate_mahal_kernel(X_train, X_train, idt_ker, gamma=g)
        K_test_idt_lin = calculate_mahal_kernel(X_test, X_train, idt_ker, gamma=g)

        K_train_idt_log = calculate_mahal_kernel(X_train_bin, X_train_bin, idt_ker, gamma=g)
        K_test_idt_log = calculate_mahal_kernel(X_test_bin, X_train_bin, idt_ker, gamma=g)


        clf_mahal_lin.fit(K_train_mahal_lin, y_train)
        clf_mahal_log.fit(K_train_mahal_log, y_train_bin)
        clf_mahal_log_c.fit(K_train_mahal_lin, y_train_bin)

        clf_idt_lin.fit(K_train_idt_lin, y_train)
        clf_idt_log.fit(K_train_idt_log, y_train_bin)

        test_errs_mahal_lin[i] = mean_squared_error(y_test, clf_mahal_lin.predict(K_test_mahal_lin))
        test_errs_mahal_log[i] = log_loss(y_test_bin, clf_mahal_log.predict_proba(K_test_mahal_log))
        test_errs_mahal_log_c[i] = log_loss(y_test_bin, clf_mahal_log_c.predict_proba(K_test_mahal_lin))
        test_errs_idt_lin[i] = mean_squared_error(y_test, clf_idt_lin.predict(K_test_idt_lin))
        test_errs_idt_log[i] = log_loss(y_test_bin, clf_idt_log.predict_proba(K_test_idt_log))

    print("Done")

    res = {"train_mahal_lin": train_errs_mahal_lin, "test_mahal_lin": test_errs_mahal_lin, "train_mahal_log": train_errs_mahal_log,
           "test_mahal_log": test_errs_mahal_log, "train_mahal_log_c": train_errs_mahal_log_c, "test_mahal_log_c": test_errs_mahal_log_c,
           "train_idt_lin": train_errs_idt_lin, "test_idt_lin": test_errs_idt_lin, "train_idt_log": train_errs_idt_log, "test_idt_log": test_errs_idt_log}

    min_val_lin_idx = np.argmin(res['train_mahal_lin'])
    min_test_lin_idx = np.argmin(res['test_mahal_lin'])
    min_val_log_idx = np.argmin(res['train_mahal_log'])
    min_test_log_idx = np.argmin(res['test_mahal_log'])
    min_val_log_c_idx = np.argmin(res['train_mahal_log_c'])
    min_test_log_c_idx = np.argmin(res['test_mahal_log_c'])

    print(f"Best gamma\nLinear Regression: Validation - gamma: {gammas[min_val_lin_idx]}, score: {res['train_mahal_lin'][min_val_lin_idx]},  Test - gamma: {gammas[min_test_lin_idx]}"
          f", score: {res['test_mahal_lin'][min_test_lin_idx]}\n"
          f"Binary Logistic Regression: Validation - gamma: {gammas[min_val_log_idx]}, score: {res['train_mahal_log'][min_val_log_idx]}  Test - gamma: {gammas[min_test_log_idx]}"
          f", score: {res['test_mahal_log'][min_test_log_idx]}\n"
          f"Contin Logistic Regression: Validation - gamma: {gammas[min_val_log_c_idx]}, score: {res['train_mahal_log_c'][min_val_log_c_idx]}  Test - gamma: {gammas[min_test_log_c_idx]}"
          f", score: {res['test_mahal_log_c'][min_test_log_c_idx]}")


    return res


def compare_kernels_log(gammas, X_train, X_test, y_train, y_test, assoc_mat, cv_fold=5, random_state=42):

    train_errs_mahal_log = np.zeros(gammas.shape[0])
    train_errs_mahal_log_c = np.zeros(gammas.shape[0])
    train_errs_log_c = np.zeros(gammas.shape[0])
    train_errs_log = np.zeros(gammas.shape[0])

    test_errs_mahal_log = np.zeros(gammas.shape[0])
    test_errs_mahal_log_c = np.zeros(gammas.shape[0])
    test_errs_log_c = np.zeros(gammas.shape[0])
    test_errs_log = np.zeros(gammas.shape[0])

    k_cv = KFold(n_splits=cv_fold, random_state=random_state, shuffle=True)

    X_train_bin = binarize_with_median(X_train)
    X_test_bin = binarize_with_median(X_test)

    for i, g in enumerate(gammas):
        val_scores = np.zeros((4, cv_fold))
        j = 0
        for train_idx, test_idx in k_cv.split(X_train):
            x_train_cv, x_train_bin_cv = X_train[train_idx], X_train_bin[train_idx]
            x_test_cv, x_test_bin_cv = X_train[test_idx], X_train_bin[test_idx]
            y_train_cv, y_test_cv = y_train[train_idx], y_train[test_idx]

            cov_mat_reg_cv = get_emp_covariance(x_train_cv, assoc_mat)
            cov_mat_bin_cv = get_emp_covariance(x_train_bin_cv, assoc_mat)

            K_train_mahal_lin_cv = calculate_mahal_kernel(x_train_cv, x_train_cv, cov_mat_reg_cv, gamma=g)
            K_test_mahal_lin_cv = calculate_mahal_kernel(x_test_cv, x_train_cv, cov_mat_reg_cv, gamma=g)

            K_train_mahal_log_cv = calculate_mahal_kernel(x_train_bin_cv, x_train_bin_cv, cov_mat_bin_cv, gamma=g)
            K_test_mahal_log_cv = calculate_mahal_kernel(x_test_bin_cv, x_train_bin_cv, cov_mat_bin_cv, gamma=g)

            clf_mahal_log = LogisticRegression()
            clf_mahal_log_c = LogisticRegression()
            clf_log_c = LogisticRegression()
            clf_log = LogisticRegression()

            clf_mahal_log.fit(K_train_mahal_log_cv, y_train_cv)
            clf_mahal_log_c.fit(K_train_mahal_lin_cv, y_train_cv)
            clf_log_c.fit(x_train_cv, y_train_cv)
            clf_log.fit(x_train_bin_cv, y_train_cv)

            val_scores[0][j] = log_loss(y_test_cv, clf_mahal_log.predict_proba(K_test_mahal_log_cv))
            val_scores[1][j] = log_loss(y_test_cv, clf_mahal_log_c.predict_proba(K_test_mahal_lin_cv))
            val_scores[2][j] = log_loss(y_test_cv, clf_log_c.predict_proba(x_test_cv))
            val_scores[3][j] = log_loss(y_test_cv, clf_log.predict_proba(x_test_bin_cv))

            j += 1

        # print(f"--- gamma val: {g} ---\nlin_mahala: {-np.mean(cv_scores_m_lin)}\nlog_mahala: {-np.mean(cv_scores_m_log)}"
        #       f"\nlin_idt:{-np.mean(cv_scores_i_lin)}\nlog_idt: {-np.mean(cv_scores_i_log)}\n")

        clf_mahal_log = LogisticRegression()
        clf_mahal_log_c = LogisticRegression()
        clf_log_c = LogisticRegression()
        clf_log = LogisticRegression()

        train_errs_mahal_log[i] = np.mean(val_scores[0])
        train_errs_mahal_log_c[i] = np.mean(val_scores[1])
        train_errs_log_c[i] = np.mean(val_scores[2])
        train_errs_log[i] = np.mean(val_scores[3])

        cov_mat_reg = get_emp_covariance(X_train, assoc_mat)
        cov_mat_bin = get_emp_covariance(X_train_bin, assoc_mat)

        K_train_mahal_lin = calculate_mahal_kernel(X_train, X_train, cov_mat_reg, gamma=g)
        K_test_mahal_lin = calculate_mahal_kernel(X_test, X_train, cov_mat_reg, gamma=g)

        K_train_mahal_log = calculate_mahal_kernel(X_train_bin, X_train_bin, cov_mat_bin, gamma=g)
        K_test_mahal_log = calculate_mahal_kernel(X_test_bin, X_train_bin, cov_mat_bin, gamma=g)


        clf_mahal_log.fit(K_train_mahal_log, y_train)
        clf_mahal_log_c.fit(K_train_mahal_lin, y_train)
        clf_log_c.fit(X_train, y_train)
        clf_log.fit(X_train_bin, y_train)

        test_errs_mahal_log[i] = log_loss(y_test, clf_mahal_log.predict_proba(K_test_mahal_log))
        test_errs_mahal_log_c[i] = log_loss(y_test, clf_mahal_log_c.predict_proba(K_test_mahal_lin))
        test_errs_log_c[i] = log_loss(y_test, clf_log_c.predict_proba(X_test))
        test_errs_log[i] = log_loss(y_test, clf_log.predict_proba(X_test_bin))
    print("Done")

    res = {"train_mahal_log": train_errs_mahal_log,"test_mahal_log": test_errs_mahal_log, "train_mahal_log_c": train_errs_mahal_log_c, "test_mahal_log_c": test_errs_mahal_log_c,
           "train_log_c": train_errs_log_c, "test_log_c": test_errs_log_c,
           "train_log": train_errs_log, "test_log": test_errs_log}

    min_val_mahal_log_idx = np.argmin(res['train_mahal_log'])
    min_test_mahal_log_idx = np.argmin(res['test_mahal_log'])
    min_val_mahal_log_c_idx = np.argmin(res['train_mahal_log_c'])
    min_test_mahal_log_c_idx = np.argmin(res['test_mahal_log_c'])
    min_val_log_c_idx = np.argmin(res["train_log_c"])
    min_test_log_c_idx = np.argmin(res["test_log_c"])
    min_val_log_idx = np.argmin(res["train_log"])
    min_test_log_idx = np.argmin(res["test_log"])

    print( f"Mahal Contin Logistic Regression: Validation - gamma: {gammas[min_val_mahal_log_c_idx]}, score: {res['train_mahal_log_c'][min_val_mahal_log_c_idx]}  Test - gamma: {gammas[min_test_mahal_log_c_idx]}"
          f", score: {res['test_mahal_log_c'][min_test_mahal_log_c_idx]}\n"
          f"Mahal Binary Logistic Regression: Validation - gamma: {gammas[min_val_mahal_log_idx]}, score: {res['train_mahal_log'][min_val_mahal_log_idx]}  Test - gamma: {gammas[min_test_mahal_log_idx]}"
          f", score: {res['test_mahal_log'][min_test_mahal_log_idx]}\n"
          f"Contin Logistic Regression (no kernel): Validation - score: {res['train_log_c'][min_val_log_c_idx]} Test - score: {res['test_log_c'][min_test_log_c_idx]}\n"
          f"Binary Logistic Regression (no kernel): Validation - score: {res['train_log'][min_val_log_idx]} Test - score: {res['test_log'][min_test_log_idx]}")


    return res

def plot_ker_comp(gammas, res, k_1, k_2, x_title, y_title, x_scale=None, y_scale=None):
    fig = figure(figsize=(12, 8))
    plt.plot(gammas, res[k_1], figure=fig, label=k_1)
    plt.plot(gammas, res[k_2], figure=fig, label=k_2)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend()
    if x_scale is not None:
        plt.xscale(x_scale)
    if y_scale is not None:
        plt.yscale(y_scale)
    plt.show()

def get_col_names(tf, genes):
    cols = ["out"]

    m = genes + 1
    for t in range(1, tf + 1):
        cols.append(f"TF{t}")
        for g in range(1, m):
            cols.append(f"Tf{t}g{g}")

    return cols

def get_assoc_mat(tf, genes):
    feats = tf + (tf * genes)
    assoc_mat = np.eye(feats, feats)
    m = genes + 1
    for t in range(0, m * tf, m):
        for g in range(t + 1, t + m):
            assoc_mat[t, g] = 1
            assoc_mat[g, t] = 1

    return assoc_mat



def save_gen_file(X, y, dir, name, tf, genes):
    X_bin = binarize_with_median(X).astype(np.int_)
    bin_arr = np.concatenate([y.reshape(-1, 1), X_bin], axis=1)
    contin_arr = np.concatenate([y.reshape(-1, 1), X], axis=1)
    cols = get_col_names(tf, genes)
    bin_df = pd.DataFrame(bin_arr, columns=cols)
    contin_df = pd.DataFrame(contin_arr, columns=cols)
    feats = X.shape[1]

    bin_path = os.path.join(dir, f"{name}_{feats}_bin.csv")
    contin_path = os.path.join(dir, f"{name}_{feats}.csv")

    bin_df.to_csv(bin_path, index=False)
    contin_df.to_csv(contin_path, index=False)


def plot_lambda_vals(gammas, train_errs, test_errs, train_penalty, test_penalty,
                            l1_vals, l2_vals, l1 = True):
    train_min_idx = np.unravel_index(np.argmin(train_errs), train_errs.shape)
    test_min_idx = np.unravel_index(np.argmin(test_errs), test_errs.shape)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
    if l1:
        ax[0, 0].plot(l1_vals, train_errs[train_min_idx[0]][:, train_min_idx[2]])
        ax[0, 0].set_xlabel(r"$\lambda_1$", fontsize=16)
        ax[0, 0].set_ylabel(r"train log_loss", fontsize=16)
        ax[0, 0].set_xscale("log")
        ax[0, 0].set_title(f"$\gamma$: {gammas[train_min_idx[0]]:.2f}, $\lambda_2$: {l2_vals[train_min_idx[2]]:.2f}")
        ax[0, 0].axvline(l1_vals[10], color="red", linestyle="--",
                         label=f"$\lambda_1$ = {l1_vals[train_min_idx[1]]: .3f}")
        ax[0, 0].legend()

        ax[0, 1].plot(l1_vals, train_penalty[train_min_idx[0]][:, train_min_idx[2]])
        ax[0, 1].set_xlabel(r"$\lambda_1$", fontsize=16)
        ax[0, 1].set_ylabel(r"$\Vert\beta\Vert^{2}$ train", fontsize=16)
        ax[0, 1].set_xscale("log")
        ax[0, 1].set_title(f"$\gamma$: {gammas[train_min_idx[0]]:.2f}, $\lambda_2$: {l2_vals[train_min_idx[2]]:.2f}")
        ax[0, 1].axvline(l1_vals[train_min_idx[1]], color="red", linestyle="--",
                         label=f"$\lambda_1$ = {l1_vals[train_min_idx[1]]: .3f}")
        ax[0, 1].legend()

        ax[1, 0].plot(l1_vals, test_errs[test_min_idx[0]][:, test_min_idx[2]])
        ax[1, 0].set_xlabel(r"$\lambda_1$", fontsize=16)
        ax[1, 0].set_ylabel(r"test log_loss", fontsize=16)
        ax[1, 0].set_xscale("log")
        ax[1, 0].set_title(f"$\gamma$: {gammas[test_min_idx[0]]:.2f}, $\lambda_2$: {l2_vals[test_min_idx[2]]:.2f}")
        ax[1, 0].axvline(l1_vals[test_min_idx[1]], color="red", linestyle="--",
                         label=f"$\lambda_1$ = {l1_vals[test_min_idx[1]]: .3f}")
        ax[1, 0].legend()

        ax[1, 1].plot(l1_vals, test_penalty[test_min_idx[0]][:, test_min_idx[2]])
        ax[1, 1].set_xlabel(r"$\lambda_1$", fontsize=16)
        ax[1, 1].set_ylabel(r"$\Vert\beta\Vert^{2}$ test", fontsize=16)
        ax[1, 1].set_xscale("log")
        ax[1, 1].set_title(f"$\gamma$: {gammas[test_min_idx[0]]:.2f}, $\lambda_2$: {l2_vals[test_min_idx[2]]:.2f}")
        ax[1, 1].axvline(l1_vals[test_min_idx[1]], color="red", linestyle="--",
                         label=f"$\lambda_1$ = {l1_vals[test_min_idx[1]]:.3f}")
        ax[1, 1].legend()
    else:
        ax[0, 0].plot(l2_vals, train_errs[train_min_idx[0]][train_min_idx[1]])
        ax[0, 0].set_xlabel(r"$\lambda_2$", fontsize=16)
        ax[0, 0].set_ylabel(r"train log_loss", fontsize=16)
        ax[0, 0].set_xscale("log")
        ax[0, 0].set_title(f"$\gamma$: {gammas[train_min_idx[0]]:.2f}, $\lambda_1$: {l1_vals[train_min_idx[1]]:.2f}")
        ax[0, 0].axvline(l2_vals[train_min_idx[2]], color="red", linestyle="--",
                         label=f"$\lambda_2$ = {l2_vals[train_min_idx[2]]: .3f}")
        ax[0, 0].legend()

        ax[0, 1].plot(l2_vals, train_penalty[train_min_idx[0]][train_min_idx[1]])
        ax[0, 1].set_xlabel(r"$\lambda_2$", fontsize=16)
        ax[0, 1].set_ylabel(r"$\mathbf{}{f}^{T}\mathbf{}{L}\mathbf{}{f}$ train", fontsize=16)
        ax[0, 1].set_xscale("log")
        ax[0, 1].set_title(f"$\gamma$: {gammas[train_min_idx[0]]:.2f}, $\lambda_1$: {l1_vals[train_min_idx[1]]:.2f}")
        ax[0, 1].axvline(l2_vals[train_min_idx[2]], color="red", linestyle="--",
                         label=f"$\lambda_2$ = {l2_vals[train_min_idx[2]]: .3f}")
        ax[0, 1].legend()

        ax[1, 0].plot(l2_vals, test_errs[test_min_idx[0]][test_min_idx[1]])
        ax[1, 0].set_xlabel(r"$\lambda_2$", fontsize=16)
        ax[1, 0].set_ylabel(r"test log_loss", fontsize=16)
        ax[1, 0].set_xscale("log")
        ax[1, 0].set_title(f"$\gamma$: {gammas[test_min_idx[0]]:.2f}, $\lambda_1$: {l1_vals[test_min_idx[1]]:.2f}")
        ax[1, 0].axvline(l2_vals[test_min_idx[2]], color="red", linestyle="--",
                         label=f"$\lambda_2$ = {l2_vals[test_min_idx[2]]: .3f}")
        ax[1, 0].legend()

        ax[1, 1].plot(l2_vals, test_penalty[test_min_idx[0]][test_min_idx[1]])
        ax[1, 1].set_xlabel(r"$\lambda_2$", fontsize=16)
        ax[1, 1].set_ylabel(r"$\mathbf{}{f}^{T}\mathbf{}{L}\mathbf{}{f}$ test", fontsize=16)
        ax[1, 1].set_xscale("log")
        ax[1, 1].set_title(f"$\gamma$: {gammas[test_min_idx[0]]:.2f}, $\lambda_1$: {l2_vals[test_min_idx[1]]:.2f}")
        ax[1, 1].axvline(l2_vals[test_min_idx[1]], color="red", linestyle="--",
                         label=f"$\lambda_2$ = {l2_vals[test_min_idx[2]]:.3f}")
        ax[1, 1].legend()


def calculate_sens_spec(beta_hat, n=44):
    idx = np.where((beta_hat > 1e-8) | (beta_hat < -1e-8))

    beta_true = np.zeros(len(beta_hat))
    beta_true[np.arange(0, n)] = 1

    beta_pred = np.zeros(len(beta_hat))
    beta_pred[idx] = 1

    tn, fp, fn, tp = confusion_matrix(beta_true, beta_pred).ravel()


    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print(f"tp: {tp}, fp: {fp}, fn: {fn}, tp: {tp}")
    return sensitivity, specificity
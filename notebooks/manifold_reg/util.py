__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

import math
import os.path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
# import autograd.numpy as np
import pandas as pd
import scipy
from sklearn.covariance import ShrunkCovariance
from matplotlib.pyplot import figure
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.metrics import mean_squared_error, log_loss, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_score, pairwise_distances
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.base import BaseEstimator
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from log_util import log_msg
import traceback


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

def get_penalty_comp_log(X, y, beta, L=None, use_coef=False):
    f = X @ beta
    m = X.shape[0]
    log_ll = np.sum((y * f) - np.log(1 + np.exp(f))) / m
    c1 = np.linalg.norm(beta, 2)

    if L is None:
        return -log_ll, c1
    else:
        if use_coef:
            c2 = beta.T @ L @ beta
        else:
            c2 = f.T @ L @ f
        return -log_ll, c1, c2

def log_loss_cp(X, y, beta):
    p = sigmoid(X @ beta)
    return log_loss(y, p)


def get_laplacian_mat(X1, X2, prec_mat, gamma, norm=False):

    K = calculate_mahal_kernel(X1, X2, prec_mat, gamma=gamma)
    # W =  np.sum(K, axis=0)
    # D = np.diag(W)
    #
    # L = D - K
    L = scipy.sparse.csgraph.laplacian(K, normed=norm)
    return L


def solve_prob(prob):
    prob.solve(solver="SCS", max_iters=10000)
    return prob.variables()[0].value



def solve_logistic_reg(X_train, X_test, y_train, y_test, l1_vals, l2_vals, gamma, assoc_mat, err_fn=log_loss_cp, lap_norm=False):

    assert l1_vals.shape[0] == l2_vals.shape[0]

    n_iter = l1_vals.shape[0]

    train_errors = np.full((n_iter, n_iter), 1e6)  # set default to 1e6 to filter out cases where the solver failed
    test_errors = np.full((n_iter, n_iter), 1e6)
    beta_vals = np.zeros((n_iter, n_iter, X_train.shape[1]))
    ll_pens = np.full((n_iter, n_iter), 1e6)
    l1_pens = np.full((n_iter, n_iter), 1e6)
    l2_pens = np.full((n_iter, n_iter), 1e6)

    prec_mat = get_emp_covariance(X_train, assoc_mat)

    L = get_laplacian_mat(X_train, X_train, prec_mat, gamma, lap_norm)

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

def apply_logisitc_reg(X_train, X_test, y_train, y_test, l1, l2, gamma, assoc_mat, err_fn=log_loss_cp, lap_norm=False):
    beta = cp.Variable(X_train.shape[1])

    prec_mat = get_emp_covariance(X_train, assoc_mat)

    L = get_laplacian_mat(X_train, X_train, prec_mat, gamma, lap_norm)

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


def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def nearPD(A, nit=100):
    n = A.shape[0]
    W = np.identity(n)
# W is the matrix used for the norm (assumed to be Identity matrix here)
# the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk

def get_psd_mat(X, tol=1e-14):
    if is_pd(X):
        return X
    else:
        regularized_X = None
        i = 14
        while i > 9:
            tol = pow(10, -i)
            regularized_X = X + np.identity(X.shape[0]) * tol
            if is_pd(regularized_X):
                break
            i += -1

    # print(f"Num iter: {14 - i}")
    return regularized_X

def get_psd_mat_2(X):
    assert X.shape[0] == X.shape[1]
    p = X.shape[0]
    l, u = np.linalg.eigh(X)
    # perturb the matrix by the smallest eigenvalue
    if l[0] > 0.0: # the matrix is already PD, no need to perturb it further
        return X
    X = X + -l[0] * np.identity(p)
    return X

# @numba.jit(parallel=True, nopython=True)
def get_mahala_dist(X, Y, cov_inv):
    D = np.zeros((X.shape[0], Y.shape[0]))
    for n in range(X.shape[0]):
        for m in range(Y.shape[0]):
            diff = X[n] - Y[m]
            D[n, m] = math.sqrt(diff @ cov_inv @ diff.T)
    return D

def get_emp_covariance(X, M=None, pinv=True):
    cov = np.cov(X, rowvar=False)
    ##sparsify the precision matrix
    # print(f"Cov rank: {np.linalg.matrix_rank(cov)}")
    # sprec = np.multiply(M, np.linalg.inv(cov))  #this one gives math erros
    mat = None
    if M is None:
        mat = cov
    else:
        mat = np.multiply(M, cov)
    if pinv:
        sprec = scipy.linalg.pinv(mat, rtol=1e-5)
    else:
        try:
            sprec = np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            print("Cov matrix is not PSD")
            mat = get_psd_mat(mat)
            sprec = np.linalg.inv(mat)

    sprec_psd = get_psd_mat(sprec)

    return sprec_psd

def calculate_mahal_kernel(X, Y, cov_inv, gamma=1.0):
    D = pairwise_distances(X, Y, metric="mahalanobis", VI=cov_inv)

    K = np.exp(-((1.0/(2.0*(np.power(gamma, 2)))) * np.power(D, 2)))

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

def is_pd(X):
    try:
        np.linalg.cholesky(X)
        return True
    except np.linalg.LinAlgError:
        return False

def shrunk_cov_score(X):
    shrinkages = np.logspace(-2, 0, 30)
    cv = GridSearchCV(ShrunkCovariance(), {"shrinkage": shrinkages}, n_jobs=-1).fit(X)
    return cv.best_estimator_, np.mean(cross_val_score(cv.best_estimator_, X))

def get_prec_mat(X, M, n_component):
    cov = np.cov(X, rowvar=False)
    if n_component is None:
        if M is None:
            return get_psd_mat(scipy.linalg.pinvh(cov))
        return get_psd_mat(M * scipy.linalg.pinvh(cov))
    l, u = np.linalg.eigh(cov)
    l = np.flip(l)
    u = np.flip(u, axis=1)
    l_p = l[:n_component]
    # print(l_p)
    l_p_inv = np.diag(1.0/(l_p))
    u_p = u[:n_component]
    prec_mat = u_p.T @ l_p_inv @ u_p
    if M is not None:
        prec_mat = np.multiply(M, prec_mat)
    prec_mat = get_psd_mat(prec_mat)
    return prec_mat

def get_prec_mat_2(cov_est, M=None, n_component=None):
    if n_component is None:
        if M is None:
            return cov_est.precision_
        return M * cov_est.precision_

    l, u = np.linalg.eigh(cov_est.covariance_)
    l = np.flip(l)
    u = np.flip(u, axis=1)
    l_p = l[:n_component]
    # print(l_p)
    l_p_inv = np.diag(1.0/(l_p))
    u_p = u[:n_component]
    prec_mat = u_p.T @ l_p_inv @ u_p
    if M is not None:
        prec_mat = np.multiply(M, prec_mat)
    # prec_mat = get_psd_mat(prec_mat)
    return prec_mat

def compare_kernels_log(gammas, X_train, X_test, y_train, y_test, assoc_mat=None, cv_fold=5, random_state=42, n_component=None, verbose=2):
    try:
        train_errs_mahal_auc = np.zeros(gammas.shape[0])
        train_errs_mahal_ll = np.zeros(gammas.shape[0])

        test_errs_mahal_auc = np.zeros(gammas.shape[0])
        test_errs_mahal_ll = np.zeros(gammas.shape[0])

        train_errs_idt_auc = np.zeros(gammas.shape[0])
        train_errs_idt_ll = np.zeros(gammas.shape[0])

        test_errs_idt_auc = np.zeros(gammas.shape[0])
        test_errs_idt_ll = np.zeros(gammas.shape[0])

        k_cv = KFold(n_splits=cv_fold, random_state=random_state, shuffle=True)
        # cov_est, sc = shrunk_cov_score(X_train)
        prec_mat = get_prec_mat(X_train, assoc_mat, n_component)
        # prec_mat = nearPD(prec_mat, 5)
        # print(f"prec_mat is PD: {is_pd(prec_mat)}, cov score: {sc}, shrinkage: {cov_est.shrinkage}")
        print(f"prec_mat is PD: {is_pd(prec_mat)}")
        for i, g in enumerate(gammas):

            # print(f"gamma - {g:.2f}")

            auc_mahal_val_scores = np.zeros(cv_fold)
            auc_idt_val_scores = np.zeros(cv_fold)
            ll_mahal_val_scores = np.zeros(cv_fold)
            ll_idt_val_scores = np.zeros(cv_fold)
            j = 0
            for train_idx, test_idx in k_cv.split(X_train):
                x_train_cv, x_test_cv = X_train[train_idx], X_train[test_idx]
                y_train_cv, y_test_cv = y_train[train_idx], y_train[test_idx]

                # weights = np.zeros(len(y_train_cv))s
                # unq = np.unique(y_train_cv, return_counts=True)
                # num_0, num_1 = unq[1][0], unq[1][1]
                # cl_0_idx = np.where(y_train_cv == 0)
                # cl_1_idx = np.where(y_train_cv == 1)
                # if num_0 > 2 * num_1:
                #     weights[cl_0_idx] = 1.0
                #     weights[cl_1_idx] = 2.0
                # elif num_1 > 2 * num_0:
                #     weights[cl_0_idx] = 2.0
                #     weights[cl_1_idx] = 1.0

                # cov_est_cv, sc_cv = shrunk_cov_score(x_train_cv)
                prec_mat_cv = get_prec_mat(x_train_cv, assoc_mat, n_component)
                # prec_mat_cv = nearPD(prec_mat_cv, 5)
                # print(f"prec_mat is PD: {is_pd(prec_mat_cv)}, cov score: {sc_cv}, shrinkage: {cov_est_cv.shrinkage}")
                print(f"prec_mat is PD: {is_pd(prec_mat_cv)}")

                K_train_mahal_cv = calculate_mahal_kernel(x_train_cv, x_train_cv, prec_mat_cv, gamma=g)
                K_test_mahal_cv = calculate_mahal_kernel(x_test_cv, x_train_cv, prec_mat_cv, gamma=g)

                K_train_idt_cv = calculate_mahal_kernel(x_train_cv, x_train_cv, np.identity(x_train_cv.shape[1]), gamma=g)
                K_test_idt_cv = calculate_mahal_kernel(x_test_cv, x_train_cv, np.identity(x_train_cv.shape[1]), gamma=g)

                # clf_mahal = lrb.LogisticRegressionBounded(C=0.0, fit_intercept=True)
                # clf_idt = lrb.LogisticRegressionBounded(C=0.0, fit_intercept=True)
                #
                # clf_mahal.fit(scipy.sparse.coo_matrix(K_train_mahal_cv), y_train_cv, sample_weight=weights)
                # clf_idt.fit(scipy.sparse.coo_matrix(K_train_idt_cv), y_train_cv, sample_weight=weights)

                clf_mahal = LogisticRegression(penalty='none', C=1e9, fit_intercept=True, class_weight="auto")
                clf_idt = LogisticRegression(penalty='none', C=1e9, fit_intercept=True, class_weight="auto")

                clf_mahal.fit(K_train_mahal_cv, y_train_cv)
                clf_idt.fit(K_train_idt_cv, y_train_cv)

                # prob_mahal = sigmoid(K_test_mahal_cv @ clf_mahal.coef_[0])
                # prob_idt = sigmoid(K_test_idt_cv @ clf_idt.coef_[0])
                prob_mahal = clf_mahal.predict_proba(K_test_mahal_cv)[:,1]
                prob_idt = clf_idt.predict_proba(K_test_idt_cv)[:,1]

                auc_mahal_val_scores[j] = roc_auc_score(y_test_cv, prob_mahal)
                auc_idt_val_scores[j] = roc_auc_score(y_test_cv, prob_idt)

                ll_mahal_val_scores[j] = log_loss(y_test_cv, prob_mahal)
                ll_idt_val_scores[j] = log_loss(y_test_cv, prob_idt)

                j += 1

            # print(f"--- gamma val: {g} ---\nlin_mahala: {-np.mean(cv_scores_m_lin)}\nlog_mahala: {-np.mean(cv_scores_m_log)}"
            #       f"\nlin_idt:{-np.mean(cv_scores_i_lin)}\nlog_idt: {-np.mean(cv_scores_i_log)}\n")

            # clf_mahal = lrb.LogisticRegressionBounded(C=0.0, fit_intercept=True)
            # clf_idt = lrb.LogisticRegressionBounded(C=0.0, fit_intercept=True)

            clf_mahal = LogisticRegression(penalty='none', C=1e9, fit_intercept=True, class_weight="auto")
            clf_idt = LogisticRegression(penalty='none', C=1e9, fit_intercept=True, class_weight="auto")

            train_errs_mahal_auc[i] = np.mean(auc_mahal_val_scores, axis=0)
            train_errs_idt_auc[i] = np.mean(auc_idt_val_scores, axis=0)

            train_errs_mahal_ll[i] = np.mean(ll_mahal_val_scores, axis=0)
            train_errs_idt_ll[i] = np.mean(ll_idt_val_scores, axis=0)


            K_train_mahal = calculate_mahal_kernel(X_train, X_train, prec_mat, gamma=g)
            K_test_mahal = calculate_mahal_kernel(X_test, X_train, prec_mat, gamma=g)
            K_train_idt = calculate_mahal_kernel(X_train, X_train, np.identity(X_train.shape[1]), gamma=g)
            K_test_idt = calculate_mahal_kernel(X_test, X_train, np.identity(X_train.shape[1]), gamma=g)

            # weights = np.zeros(len(y_train))
            # unq = np.unique(y_train, return_counts=True)
            # num_0, num_1 = unq[1][0], unq[1][1]
            # tot = num_0 + num_1
            # cl_0_idx = np.where(y_train == 0)
            # cl_1_idx = np.where(y_train == 1)
            # if num_0 > 2 * num_1:
            #     weights[cl_0_idx] = 1.0
            #     weights[cl_1_idx] = 2.0
            # elif num_1 > 2 * num_0:
            #     weights[cl_0_idx] = 2.0
            #     weights[cl_1_idx] = 1.0
            # clf_mahal.fit(scipy.sparse.coo_matrix(K_train_mahal), y_train, sample_weight=weights)
            # clf_idt.fit(scipy.sparse.coo_matrix(K_train_idt), y_train, sample_weight=weights)

            # prob_mahal = sigmoid(K_test_mahal @ clf_mahal.coef_[0])
            # prob_idt = sigmoid(K_test_idt @ clf_idt.coef_[0])

            clf_mahal.fit(K_train_mahal, y_train)
            clf_idt.fit(K_train_idt, y_train)

            prob_mahal = clf_mahal.predict_proba(K_test_mahal)[:,1]
            prob_idt = clf_idt.predict_proba(K_test_idt)[:,1]

            test_errs_mahal_auc[i] = roc_auc_score(y_test, prob_mahal)
            test_errs_idt_auc[i] = roc_auc_score(y_test, prob_idt)

            test_errs_mahal_ll[i] = log_loss(y_test, prob_mahal)
            test_errs_idt_ll[i] = log_loss(y_test, prob_idt)

        # print("Done")


        return train_errs_mahal_auc, train_errs_mahal_ll, test_errs_mahal_auc, test_errs_mahal_ll, \
               train_errs_idt_auc, train_errs_idt_ll, test_errs_idt_auc, test_errs_idt_ll
    except Exception as e:
        log_msg("Oh, no! An error occurred. Here is the error message")
        log_msg(traceback.format_exc())

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

def get_assoc_mat(tf, genes, corr=1, bias=False):
    feats = tf + (tf * genes)
    assoc_mat = np.eye(feats, feats)
    m = genes + 1
    for t in range(0, m * tf, m):
        for g in range(t + 1, t + m):
            assoc_mat[t, g] = 1
            assoc_mat[g, t] = 1
    if bias:
        zero_col = np.zeros((assoc_mat.shape[0], 1))
        zero_row = np.zeros((1, assoc_mat.shape[0] + 1))
        assoc_mat = np.hstack((zero_col, assoc_mat))
        return np.vstack((zero_row, assoc_mat))
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
    idx = np.where((beta_hat > 1e-5) | (beta_hat < -1e-5))

    beta_true = np.zeros(len(beta_hat))
    beta_true[np.arange(0, n)] = 1

    beta_pred = np.zeros(len(beta_hat))
    beta_pred[idx] = 1

    tn, fp, fn, tp = confusion_matrix(beta_true, beta_pred).ravel()


    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print(f"tp: {tp}, fp: {fp}, fn: {fn}, tp: {tp}")
    return sensitivity, specificity

def preprocess_data(X_train, X_test):
    # scale the X values to standard normal distribution
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    #change 0s to -1 so the y is element of {-1, 1}
    #
    # y_train_t = y_train.copy()
    # idx_train = np.where(y_train_t == 0)
    # y_train_t[idx_train] = -1
    #
    # y_test_t = y_test.copy()
    # idx_test = np.where(y_test_t == 0)
    # y_test_t[idx_test] = -1

    return X_train_s, X_test_s

class KernelLogisiticRegression(BaseEstimator):

    def __init__(self, gamma=1.0, n_components=None,  assoc_mat=None,
                    shrink_cov=False, fit_intercept=True, penalty="none", identity=False, alpha=1.0):
        """
        Constructs a model that performs Logisitic regression in the feature space indudced by a kernel
        :param gamma: The rbf kernel paramter
        :param n_components: The number of components to retain to construct the inverse of the covariance matrix
        :param assoc_mat: The adjancency matrix to use a mask over precision matrix
        :param shrink_cov: Whether to perform ridge-regression in estimating the covariance matrix
        :param fit_intercept: Logistic Regression parameter - see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        :param penalty: Logistic Regression parameter - see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        :param identity: Whether to use an identity matrix for mahalanobis distance
        """

        self.gamma = gamma
        self.n_components = n_components
        self.assoc_mat = assoc_mat
        self.shrink_cov = shrink_cov
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.identity = False
        self.alpha = alpha

    def get_params(self, deep=True):
        return {"gamma": self.gamma, "n_components": self.n_components, "assoc_mat": self.assoc_mat,
                    "shrink_cov": self.shrink_cov, "fit_intercept": self.fit_intercept,
                    "penalty": self.penalty, "identity": self.identity, "alpha": self.alpha}

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)

        return self

    def fit(self, X, y):
        self.X_ = X
        if not self.identity:
            self.prec_mat_ = self._get_prec_mat(self.X_, self.assoc_mat, self.n_components)
        else:
            self.prec_mat_ = np.identity(self.X_.shape[1])

        self.K_ = calculate_mahal_kernel(self.X_, self.X_,  self.prec_mat_, self.gamma)
        self.clf_ = LogisticRegression(fit_intercept=self.fit_intercept, penalty=self.penalty, class_weight='auto')
        self.clf_.fit(self.K_, y)
        self.coef_ = self.clf_.coef_
        self.intercept_ = self.clf_.intercept_

        return self

    def predict(self, X):
        K = calculate_mahal_kernel(X, self.X_, self.prec_mat_, self.gamma)
        return self.clf_.predict(K)

    def predict_proba(self, X):
        K = calculate_mahal_kernel(X, self.X_, self.prec_mat_, self.gamma)
        return self.clf_.predict_proba(K)[:,1]

    def score(self, X, y):
        probs = self.predict_proba(X)
        return roc_auc_score(y, probs)

    def decision_function(self, X):
        K = calculate_mahal_kernel(X, self.X_, self.prec_mat_, self.gamma)
        scores = K @ self.coef_.T + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores

    def _shrunk_cov_score(self, X):
        shrinkages = np.logspace(-2, 0, 30)
        cv = GridSearchCV(ShrunkCovariance(), {"shrinkage": shrinkages}, n_jobs=-1).fit(X)
        return cv.best_estimator_

    def _get_prec_mat(self, X, M, n_component):
        est = None
        if self.shrink_cov:
            est = self._shrunk_cov_score(X)
            cov = est.covariance_
        else:
            cov = np.cov(X, rowvar=False)
        if n_component is None:
            if M is None:
                return self._get_psd_mat(scipy.linalg.pinvh(cov))
            else: return self._get_psd_mat(np.multiply((self.alpha * M), scipy.linalg.pinvh(cov)))

        l, u = np.linalg.eigh(cov)
        l = np.flip(l)
        u = np.flip(u, axis=1)
        l_p_inv = np.diag(1.0 / (l[:n_component]))
        prec_mat = u[:,:n_component] @ l_p_inv @ u[:,:n_component].T
        if M is not None:
            prec_mat = np.multiply((self.alpha * M), prec_mat)
        prec_mat = self._get_psd_mat(prec_mat)
        return prec_mat

    def _get_psd_mat(self, X):
        assert X.shape[0] == X.shape[1]
        p = X.shape[0]
        l, u = np.linalg.eigh(X)
        # perturb the matrix by the smallest eigenvalue
        if l[0] > 0.0: # the matrix is already PD, no need to perturb it further
            return X
        X = X + -l[0] * np.identity(p)
        return X

def assign_cols(X, append_y=True):
    cols = []
    if append_y:
        for i in range(X.shape[1] - 1):
            cols.append(f"f{i + 1}")

        cols.append("y")
    else:
        for i in range(X.shape[1]):
            cols.append(f"f{i + 1}")
    X.columns = cols

def load_bmm_files(parent_dir):
    net_dir = os.path.join(parent_dir, "net")
    feat_dir = os.path.join(parent_dir, "feats")
    data_dir = os.path.join(parent_dir, "data")

    net_dfs = []
    data_dfs = []
    feat_ls = []
    seed_str = ""

    with open(os.path.join(parent_dir, "rand_seeds.txt"), "r") as fp:
        seed_str = fp.readline().strip()

    seeds = [int(s) for s in seed_str.split(',')]

    for s in seeds:
        data_df = pd.read_csv(os.path.join(data_dir, f"data_bm_{s}.csv"), header=None)
        net_df = pd.read_csv(os.path.join(net_dir, f"feat_net_{s}.csv"), header=None)
        with open(os.path.join(feat_dir, f"feats_{s}.txt"), "r") as fp:
            feats_str = fp.readline().strip()

        feats = [int(f) for f in feats_str.split(',')]

        assign_cols(data_df)

        data_dfs.append(data_df)
        net_dfs.append(net_df)
        feat_ls.append(feats)


    return seeds, data_dfs, net_dfs, feat_ls

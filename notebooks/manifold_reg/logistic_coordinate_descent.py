__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

import autograd.numpy as anp
import numpy as np
from autograd import grad, hessian, elementwise_grad as egrad, jacobian
import pandas as pd
import scipy
import math
from sklearn.metrics import log_loss

def sigmoid(b, X):
    return 1/(1 + anp.exp(-anp.dot(X, b)))

def soft_threshold(z, gamma):
    if z > 0 and gamma < abs(z):
        return z - gamma
    if z < 0 and gamma < abs(z):
        return z + gamma
    return 0

def logistic_loss(b, X, y):
    y_t = y.copy()
    y_t[y_t == 0] = -1
    f = y_t * (X @ b)
    return anp.sum(anp.log(1 + anp.exp(-f)))


def lasso_pen(b):
    return anp.sum(anp.abs(b))

def quad_pen(b, L):
    return b.T @ L @ b

def log_loss_cp(beta, X, y):
    p = sigmoid(beta, X)
    return log_loss(y, p)

def obj_fn(beta, X, y, L, l, alpha):
    return log_loss_cp(beta, X, y) + l*((1 - alpha)*quad_pen(beta, L) + alpha*lasso_pen(beta))

def elementwise_hess(fun, argnum=0):
    sum_latter_dims = lambda x: np.sum(x.reshape(x.shape[0], -1), 1)
    def sum_grad_output(*args, **kwargs):
        return sum_latter_dims(egrad(fun)(*args, **kwargs))
    return jacobian(sum_grad_output, argnum)

def approximate_prob(p):
    idx_1 = np.where(1 - p <= 1e-5)
    idx_0 = np.where(p <= 1e-5)
    p[idx_1] = 1.0
    p[idx_0] = 0.0
    return p

def grad_logisitic(b, X, y):
    u_1 = sigmoid(b, X)
    grad = X.T @ (u_1 - y)
    return grad


def coordinate_descent(X_train, X_test, y_train, y_test, L, lambda_vals, alphas, tol=1e-3, max_iter=100):

    p = X_train.shape[1]
    betas = np.zeros((len(lambda_vals), len(alphas), p))
    test_losses = np.zeros((len(lambda_vals), len(alphas)))
    test_errs = np.zeros((len(lambda_vals), len(alphas)))
    m_iter = -1
    for i, l in enumerate(lambda_vals):
        print(f"lambda: {l}")
        for a, alpha in enumerate(alphas):
            losses = []
            beta_ = np.zeros((max_iter, p))
            if i > 0 and a == 0:
                beta_[0] = betas[i - 1, a]
            if i > 0 and a > 0:
                beta_[0] = betas[i, a - 1]
            for k in range(1, max_iter + 1):
                for j in range(p):
                    beta_cp = np.zeros(p)
                    beta_cp[j] = beta_[k-1, j]
                    u_1 = sigmoid(beta_cp, X_train)
                    u_1 = approximate_prob(u_1)
                    grad = X_train.T @ (u_1 - y_train)
                    quad_grad_j = egrad(quad_pen)(beta_cp, L)[j]
                    # S = np.diag(u_1*(1 - u_1))
                    # hess = X.T @ S @ X
                    if i == 0 and k == 1:
                        eta = 0.1
                    else:
                        pk = np.ones(p)
                        # pk[j] = 1
                        eta = scipy.optimize.line_search(logistic_loss, grad_logisitic, beta_cp, pk, gfk=grad, args=(X_train, y_train), maxiter=100)[0]

                    if eta is None: eta = 0.1
                    # u_1 = approximate_prob(u_1)
                    # beta_val = beta_cp[j] - (np.linalg.inv(hess) @ grad)[j]
                    beta_val = beta_cp[j] - eta * grad[j]
                    st = soft_threshold(beta_val, l * alpha)
                    t = st/(1 + l * (1 - alpha) * quad_grad_j)
                    beta_[k - 1, j] = t

                loss_k = obj_fn(beta_[k - 1], X_train, y_train, L, l, alpha)
                losses.append(loss_k)
                if k > 1 and (losses[-1] - loss_k) < tol:
                    m_iter = k - 1
                    break
            # print(beta_[-1])
            test_losses[i, a] = obj_fn(beta_[m_iter], X_test, y_test, L, l, alpha)
            test_errs[i, a] = log_loss_cp(beta_[m_iter], X_test, y_test)
            betas[i, a] = beta_[m_iter]
            m_iter = -1
    return test_errs, betas, test_losses


def apply_coordinate_descent(X_train, X_test, y_train, y_test, L, l, alpha, tol=1e-3, max_iter=100):

    p = X_train.shape[1]
    betas = np.zeros(p)
    m_iter = -1
    beta_ = np.zeros((max_iter, p))
    losses = []
    for k in range(1, max_iter + 1):
        for j in range(p):
            beta_cp = np.zeros(p)
            beta_cp[j] = beta_[k-1, j]
            u_1 = sigmoid(beta_cp, X_train)
            u_1 = approximate_prob(u_1)
            grad = X_train.T @ (u_1 - y_train)
            quad_grad_j = egrad(quad_pen)(beta_cp, L)[j]
            # S = np.diag(u_1*(1 - u_1))
            # hess = X.T @ S @ X
            if k == 1:
                eta = 0.1
            else:
                pk = np.ones(p)
                # pk[j] = 1
                eta = scipy.optimize.line_search(logistic_loss, grad_logisitic, beta_cp, pk, gfk=grad, args=(X_train, y_train), maxiter=100)[0]

            if eta is None: eta = 0.1
            # u_1 = approximate_prob(u_1)
            # beta_val = beta_cp[j] - (np.linalg.inv(hess) @ grad)[j]
            beta_val = beta_cp[j] - eta * grad[j]
            st = soft_threshold(beta_val, l * alpha)
            t = st/(1 + l * (1 - alpha) * quad_grad_j)
            beta_[k - 1, j] = t

        loss_k = obj_fn(beta_[k - 1], X_train, y_train, L, l, alpha)
        losses.append(loss_k)
        if k > 1 and (losses[-1] - loss_k) < tol:
            m_iter = k - 1
            break
    print(losses[-1])
    test_losses = obj_fn(beta_[m_iter], X_test, y_test, L, l, alpha)
    test_errs = log_loss_cp(beta_[m_iter], X_test, y_test)

    return test_errs, beta_[m_iter], test_losses
import sys
from optparse import OptionParser

import scipy
import numpy as np
from autograd import jacobian, hessian
from scipy import sparse
from scipy.special import expit
from libc.stdio cimport printf

from libc.math cimport exp
from libc.math cimport log
from libc.math cimport abs as c_abs
cimport numpy as np
from scipy.linalg.cython_blas cimport dgemm, ddot
from cython cimport boundscheck, wraparound

DTYPE = np.double

'''
Logistic regression - where the quadratic penalty is on the probabilities
'''

class LogisticRegressionBounded:

    def __init__(self, C=0.0, D=0.0, fit_intercept=False, lower=None, upper=None, do_elimination=True):
        self._C = C
        self._D = D
        if lower is None:
            self._lower = -np.inf
        else:
            self._lower = lower
        if upper is None:
            self._upper = np.inf
        else:
            self._upper = upper
        self._fit_intercept = fit_intercept
        self._do_elimination = do_elimination
        self._w = None          # model weights
        self.coef_ = None
        self.intercept_ = 0.0        
        self._L = None          # loss
        self._Q = None          # quadratic penalty
        self._A = None          # Laplacian matrix
        self._R = None          # regularization penalty 
        self._exp_nyXw = None   # stored vector of exp(-yXw) values
        self._probs = None      # stored vector of 1/(1+exp(-yXw)) values        
        self._active = None     # vector of active variables
        self._v = None          
        self._M = 0

    def decision_function(self, X):
        return X.dot(self.coef_[0]) + self.intercept_

    def fit(self, X, y, A, sample_weight=None, beta=0.9, sigma=0.01, tol=1e-5, min_epochs=2, max_epochs=200,
            init_w=None, verbose=0, randomize=False, max_ls_steps=2000):
        """
        Coordinate descent with Newton directions for L1-regularized logistic regression
        :param X: n x p feature matrix
        :param y: vector of labels in {0, 1}
        :paran A: matrix for quadratic penalty
        :param max_iter:
        :return:
        """

        n_items, n_features = X.shape
        assert sparse.issparse(X)
        self._A = A
        # add an intercept term if desired
        if self._fit_intercept:
            X = sparse.hstack([np.ones((n_items, 1)), X])
            n_features += 1

        if sample_weight is None:
            sample_weights = np.ones(n_items)
        else:
            sample_weights = sample_weight
            assert len(sample_weights) == n_items

        # change labels to {-1, +1}
        y = np.array(y, dtype=np.int32)
        y[y == 0] = -1
        # premultiply y * X
        yX = X.multiply(y.reshape((n_items, 1))).tocsc()
        # convert sparse matrix to a set of vectors and indices
        yX_j_starts = np.zeros(n_features, dtype=np.int32)
        yX_j_lengths = np.zeros(n_features, dtype=np.int32)
        yX_rows = []
        yX_vals = []
        index = 0
        for j in range(n_features):                
            yX_j_coo = yX[:, j].tocoo()
            yX_j_length = len(yX_j_coo.data)
            yX_j_starts[j] = index
            yX_j_lengths[j] = yX_j_length
            for row, val in zip(yX_j_coo.row, yX_j_coo.data):
                yX_rows.append(row)
                yX_vals.append(val)
            index += yX_j_length
        yX_rows = np.array(yX_rows, dtype=np.int32)
        yX_vals = np.array(yX_vals)

        # initialize coefficients
        if init_w is None:
            self._w = np.zeros(n_features)
        else:
            self._w = init_w.copy()
            assert self._w.shape[0] == n_features

        # initialize all remaining variables
        self._v = np.zeros(n_features)
        self._active = np.ones(n_features, dtype=np.int32)
        self._M = 0
        self._exp_nyXw = np.exp(-yX.dot(self._w))
        self._probs = 1.0 / (1.0 + self._exp_nyXw)
        self._Q = self._D * quad_form(self._probs, self._A)
        print("Initial Background pen %0.5f - reloaded v%d" % (self._Q, 2))
        self._R = self._C * np.sum(np.abs(self._w))
        self._L = np.sum(sample_weights * np.log(1.0 + self._exp_nyXw))

        order = np.array(np.arange(n_features), dtype=np.int32)

        print("C=%0.5f, D=%0.5f" % (self._C, self._D))

        for k in range(max_epochs):
            if randomize:
                np.random.shuffle(order)

            delta, ls_steps, L, Q, R = sparse_update(n_items, n_features, self._fit_intercept, self._C, self._D, beta, sigma, self._L, self._Q, self._R, self._probs, self._exp_nyXw, self._w, self._lower, self._upper, yX_j_starts, yX_j_lengths, yX_rows, yX_vals, sample_weights, order, self._M, self._v, self._active, max_ls_steps, self._A)
            self._L = L
            self._Q = Q
            self._R = R

            # update the threshold for eliminating variables on the next iteration
            if self._do_elimination and k > 0:
                M = np.max(self._v / k)

            w_sum = np.sum(np.abs(self._w))
            if w_sum > 0:
                rel_change = delta / w_sum
            else:
                rel_change = 0.0
            if verbose > 1:
                print("epoch %d, delta=%0.5f, rel_change=%0.5f, ls_steps=%d, avg. log_loss=%0.5f, "
                      "bp_pen=%0.5f, lasso_pen=%0.5f" % (k, delta, rel_change, ls_steps, (L/n_items), Q, R))
            if rel_change < tol and k >= min_epochs - 1:
                if verbose > 0:
                    print("relative change below tolerance; stopping after %d epochs" % k)
                break

        if k == max_epochs:
            print("Maximum epochs exceeded; stopping after %d epochs" % k)

        if self._fit_intercept:
            self.intercept_ = [self._w]
            self.coef_ = self._w[1:].reshape((1, n_features-1))
        else:
            self.intercept_ = [0]
            self.coef_ = self._w


    def predict(self, X):
        return np.array(X.dot(self.coef_[0]) + self.intercept_ > 0, dtype=int)

    def predict_proba(self, X):
        n_items, n_features = X.shape
        if self._fit_intercept:
            X = sparse.hstack([np.ones((n_items, 1)), X])
            n_features += 1
        probs = np.zeros([n_items, 2])
        prob_pos = expit(X.dot(self._w))
        probs[:, 1] = prob_pos
        probs[:, 0] = 1.0 - prob_pos
        return probs


cdef sparse_update(int n_items, int n_features, int fit_intercept, double C, double D, double beta, double sigma, double L, double Q, double R, double[:] probs, double[:] exp_nyXw, np.ndarray[np.float_t, ndim=1] w, double lower, double upper, int[:] yX_j_starts, int[:] yX_j_lengths, int[:] yX_rows, double[:] yX_vals, double[:] sample_weights, int[:] order, double M, double[:] v, int[:] active, int max_ls, double[:,:] A):

        cdef int i = 0
        cdef int j
        cdef double running_abs_change = 0
        cdef int running_ls_steps = 0
        cdef double[:] exp_nyXw_new = np.zeros(n_items)
        cdef int ls_steps
        cdef double change
        cdef int is_active
        cdef int is_bias = 0
        while i < n_items:
            exp_nyXw_new[i] = exp_nyXw[i]
            i += 1

        i = 0
        while i < n_features:
            j = order[i]
            if active[j] > 0:
                # note if this is the bias term or not, to avoid bounding it
                if j == 0 and fit_intercept:
                    is_bias = 1
                else:
                    is_bias = 0
                ls_steps, change, L, R, Q, v_j, is_active = sparse_update_one_coordinate(C, D, beta, sigma, L, Q, R, probs, exp_nyXw, exp_nyXw_new, w[j], lower, upper, yX_j_starts[j], yX_j_lengths[j], yX_rows, yX_vals, sample_weights, M, is_bias, max_ls, j, A)
                if c_abs(change) > 0:
                    w[j] += change
                v[j] = v_j
                active[j] = is_active

                running_abs_change += np.abs(change)
                running_ls_steps += ls_steps
            i += 1

        return running_abs_change, running_ls_steps, L, Q, R



cdef sparse_update_one_coordinate(double C, double D, double beta, double sigma, double L, double Q, double R, double[:] probs, double[:] exp_nyXw, double[:] exp_nyXw_new, double w_j, double lower, double upper, int yX_j_start, int yX_j_length, int[:] yX_rows, double[:] yX_vals, double[:] sample_weights, double M, int is_bias, int max_ls, int j, double[:,:] A):

    cdef int index
    cdef int i
    cdef double f_val = L + Q + R
    cdef double thresh

    cdef double d  # base Newton step
    cdef double a = 1.0  # my name for lambda
    cdef double line_steps = 0
    cdef double v_j
    cdef int active = 1
    cdef double base_L = L
    cdef double base_Q = Q
    cdef double base_R = R
    cdef double[:] prob_t = np.zeros(probs.shape[0]) #hold temporary probability values when a single feature is used

    # compute the gradient and Hessian elements
    cdef double g = compute_grad_j(D, probs,  yX_j_start, yX_j_length, yX_rows, yX_vals, sample_weights, A)
    cdef double h = compute_hessian_element(D, probs, yX_j_start, yX_j_length, yX_rows, yX_vals ,sample_weights, A)

    if M > 0:
        # if w is 0 and the gradient is small, eliminate the variable from the active set
        if w_j == 0 and -1 + M < g < 1 - M:
            #print("Eliminating %d" % j)                
            active = 0
            v_j = 0
            a = 0.0
            # print("Eliminating a variable")
            return line_steps, a, L, R, v_j, active

    # compute a new value for updating self._M
    if w_j > 0:
        v_j = c_abs(g + 1)
    elif w_j < 0:
        v_j = c_abs(g - 1)
    else:
        v_j = g - 1
        if -1 - g > v_j:
            v_j = -1 - g
        if 0 > v_j:
            v_j = 0

    # do soft-thresholding
    if g + 1.0 <= h * w_j:
        d = -(g + 1.0) / h
    elif g - 1.0 >= h * w_j:
        d = -(g - 1.0) / h
    else:
        d = -w_j

    # check upper and lower limits (except for bias), and set max step accordingly
    if is_bias==0:
        if w_j + d < lower:
            diff = lower - w_j
            a = diff / d
        if w_j + d > upper:
            diff = upper - w_j
            a = diff / d

    # unless we've hit a bound, use line search to find how far to move in this direction
    if a > 0 and c_abs(d) > 0:
        # set up the threshold for convergence
        thresh = sigma * (g * d + C * c_abs(w_j + d) - C * c_abs(w_j))
        # remove the current weight from the stored 1-norm of weights
        base_R = base_R - C * c_abs(w_j)

        # also remove the influence of the relevant parts of exp(-yXw)
        i = yX_j_start
        while i < yX_j_start + yX_j_length:
            index = yX_rows[i]
            base_L = base_L - sample_weights[index] * log(1.0 + exp_nyXw[index])
            prob_t[index] = 1.0/(1 + exp_nyXw[index])
            i += 1

        base_Q = base_Q - D * quad_form(prob_t, A)
        # do line search
        #f_new, a, i, exp_nyXw = self._line_search(yX_j, d, self._w[j], R_minus_w_j, a, thresh)
        line_steps = sparse_line_search(C, D, g, h, f_val, exp_nyXw, exp_nyXw_new, yX_j_start, yX_j_length, yX_rows, yX_vals, sample_weights, d, w_j, base_L, base_Q, base_R, a, beta, thresh, max_ls, j, A)
        a = a * (beta ** line_steps)

        w_j += a * d

        # update the objective pieces

        # base_R = base_R + C * c_abs(w_j + a * d) I think this is a bug
        base_R = base_R + C * c_abs(w_j)

        i = yX_j_start
        while i < yX_j_start + yX_j_length:
            index = yX_rows[i]
            base_L = base_L + sample_weights[index] * log(1.0 + exp_nyXw_new[index])
            i += 1
            # also update the relevant values of exp(-yXw) and 1/(1+exp(-yXw))
            exp_nyXw[index] = exp_nyXw_new[index]
            probs[index] = 1.0 / (1.0 + exp_nyXw_new[index])

        base_Q = base_Q + D * quad_form(probs, A)

    return line_steps, a * d, base_L, base_Q, base_R, v_j, active

cdef double compute_grad_j(double D, double[:] probs, int yX_j_start, int yX_j_length, int[:] yX_rows, double[:] yX_vals, double[:] sample_weights, double[:,:] A):
    cdef double[:] p = np.zeros(probs.shape[0])
    cdef double g1 = 0.0
    cdef double g2 = 0.0

    fast_mat_vec_prod_1(A, probs, p)

    cdef int i = yX_j_start
    cdef int index
    while (i < yX_j_start + yX_j_length):
        index = yX_rows[i]
        u = probs[index]
        g1 += sample_weights[index] * yX_vals[i] * (u - 1.0)
        g2 += 2 * sample_weights[index] * p[index] * u * (1 - u) * yX_vals[i]
        i += 1

    return g1 + D * g2

cdef double compute_hessian_element(double D, double[:] probs, int yX_j_start, int yX_j_length, int[:] yX_rows, double[:] yX_vals, double[:] sample_weights, double[:, :] A):
    cdef double h1 = 0.0
    cdef double h2 = 0.0
    cdef int i = yX_j_start
    cdef int index
    while (i < yX_j_start + yX_j_length):
        index = yX_rows[i]
        h1 += sample_weights[index] * probs[index] * (1.0 - probs[index]) * yX_vals[i] * yX_vals[i]
        i += 1

    h2 = D * compute_hessian_quad_form(probs, yX_j_start, yX_j_length, yX_rows, yX_vals, A)
    return h1 + h2



cdef double sparse_line_search(double C, double D, double g, double h, double f_val, double[:] exp_nyXw_orig, double[:] exp_nyXw, int yX_j_start, int yX_j_length, int[:] yX_rows, double[:] yX_vals, double[:] sample_weights, double d, double prev_w_j, double base_L, double base_Q, double base_R, double a, double beta, double thresh, int max_ls, j, double[:,:] A):
    cdef line_steps = 0
    cdef double orig_a = a
    cdef double step = a * d
    cdef double w_j = prev_w_j + step

    cdef double L = base_L
    cdef double Q = base_Q
    cdef double R = base_R
    cdef double[:] prob_t = np.zeros(A.shape[0])

    cdef int index
    cdef int i = yX_j_start
    while i < yX_j_start + yX_j_length:
        index = yX_rows[i]
        exp_nyXw[index] = exp_nyXw_orig[index] * exp(-step * yX_vals[i])
        L = L +  sample_weights[index] * log(1.0 + exp_nyXw[index])
        prob_t[index] = 1.0/(1.0 + exp_nyXw[index])
        i += 1

    Q = base_Q + D*quad_form(prob_t, A)
    R = base_R + C * abs(w_j)
    cdef double f_new = L + Q + R
    cdef int count = 0
    # check for convergence (and also set an upper limit)
    while f_new - f_val > a * thresh and count < max_ls:
        line_steps += 1        
        a = a * beta
        step = a * d
        w_j = prev_w_j + step

        i = yX_j_start
        L = base_L
        while i < yX_j_start + yX_j_length:
            index = yX_rows[i]
            exp_nyXw[index] = exp_nyXw_orig[index] * exp(-step * yX_vals[i])
            L = L +  sample_weights[index] * log(1.0 + exp_nyXw[index])
            prob_t[index] = 1.0 / (1.0 + exp_nyXw[index])
            i += 1

        Q = base_Q + D*quad_form(prob_t, A)
        R = base_R + C * abs(w_j)
        f_new = L + Q + R
        count += 1

    return line_steps


@boundscheck(False)
@wraparound(False)
cpdef int fast_dgemm(double[:, :] a, double[:, :] b, double[:, :] c, double alpha=1.0, double beta=0.0,
            int ta = 0, int tb = 0) nogil except -1:
    cdef:
        char* transa
        char* transb
        int m, n, k, lda, ldb
        double *a0=&a[0, 0]
        double *b0=&b[0, 0]
        double *c0=&c[0, 0]

    if ta == 0:
        transa = 'n'
        m = a.shape[0]
        k = a.shape[1]
        lda = m
    elif ta == 1:
        transa = 't'
        m = a.shape[1]
        k = a.shape[0]
        lda = k

    if tb == 0:
        transb = 'n'
        ldb = k
        n = b.shape[1]
    elif tb == 1:
        transb = 't'
        n = b.shape[0]
        ldb = n

    dgemm(transa, transb, &m, &n, &k, &alpha, a0, &lda, b0, &ldb, &beta, c0,  &m)

    return 0

@boundscheck(False)
@wraparound(False)
cpdef double fast_dot_prod(double[:] a, double[:] b, double beta=0.0):
    cdef:
        double *a0=&a[0]
        double *b0=&b[0]
        int k
        int n

    n = a.shape[0]
    k = 1

    return ddot(&n, a0, &k, b0, &k)

@boundscheck(False)
@wraparound(False)
cpdef double quad_grad_j(int j, double[:] w, double[:, :] A):
    assert A.shape[0] == A.shape[1]
    return 2 * fast_dot_prod(A[j], w)


@boundscheck(False)
@wraparound(False)
cpdef int fast_mat_vec_prod_1(double[:,:] a, double[:] b, double[:] c, double beta=0.0) nogil except -1:

    cdef:
        double *a0
        double *b0 = &b[0]
        int k
        int n

    n = a.shape[0]
    k = 1

    for i in range(n):
        a0 = &a[i, 0]
        c[i] = ddot(&n, a0, &k, b0, &k)

    return 0

@boundscheck(False)
@wraparound(False)
cpdef int fast_mat_vec_prod_2(double[:,:] a, double[:] b, double[:] c, double beta=0.0) nogil except -1:

    cdef:
        double *a0
        double *b0 = &b[0]
        int k
        int n
        double s

    n = a.shape[0]
    k = 1

    for i in range(n):
        s = 0
        for j in range(n):
            s += b[j] * a[j, i]
        c[i] = s

    return 0

@boundscheck(False)
@wraparound(False)
cpdef double quad_form(double[:] b, double[:,:] A):

    cdef double[:] res1 = np.zeros(b.shape[0])
    fast_mat_vec_prod_2(A, b, res1)
    return fast_dot_prod(res1, b)

@boundscheck(False)
@wraparound(False)
cpdef double compute_hessian_quad_form(double[:] probs, int yX_j_start, int yX_j_length, int[:] yX_rows, double[:] yX_vals, double[:, :] A):

    cdef double h2 = 0.0
    cdef int lim = yX_j_start + yX_j_length
    cdef double[:] p = np.zeros(probs.shape[0])
    cdef double[:,:] m = np.zeros((1, yX_j_length))
    cdef double[:,:] r = np.zeros((1, yX_j_length))
    cdef double[:,:] D = np.zeros((probs.shape[0], probs.shape[0]))
    cdef double[:,:] E = np.zeros((probs.shape[0], probs.shape[0]))
    cdef double[:, :] R1 = np.zeros((probs.shape[0], probs.shape[0]))
    cdef double[:, :] R2 = np.zeros((probs.shape[0], probs.shape[0]))
    cdef double[:, :] R3 = np.zeros((probs.shape[0], probs.shape[0]))
    cdef int i = yX_j_start
    cdef int index

    fast_mat_vec_prod_1(A, probs, p)

    while (i < lim):
        index = yX_rows[i]
        u = probs[index] * (1 - probs[index])
        D[index, index] = u
        E[index, index] = p[index] * u * (1 - 2 * probs[index])
        i += 1

    m[0] = yX_vals[yX_j_start:lim]
    fast_dgemm(D, A, R1)
    fast_dgemm(R1, D, R2)

    for k in range(probs.shape[0]):
        for j in range(probs.shape[0]):
            R3[k, j] = E[k, j] + R2[k, j]

    fast_dgemm(m, R3, r)
    h2 = 2 * fast_dot_prod(r[0], m[0])

    return h2
from optparse import OptionParser

import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator

"""
NOTE: this python code was used in development and does not use cython. However, it is ver slow, and not up to date
 with all of the options available in the cython code (lrb.pyc).
"""

# deal with the @profile wrappers below
try:
    profile
except NameError:
    profile = lambda x: x


class CDN(BaseEstimator):

    def __init__(self, C=1.0, D=1.0, beta=0.5, sigma=0.01, lower=None, upper=None, do_elimination=True):
        self._C = C
        self._D = D
        self._beta = beta
        self._sigma = sigma
        self._w = None
        self._R = None
        self._exp_nyXw = None
        self._expits = None
        self._f_val = None
        self._changed = True
        self._g = None
        self._H = None
        self._lower = lower
        self._upper = upper
        self._do_elimination = do_elimination
        # variables for eliminating inactive weights
        self._active = None
        self._v = None
        self._M = None


    @profile
    def fit(self, X, y, L, tol=1e-3, min_epochs=2, max_epochs=200, init_w=None, verbose=0, randomize=False, max_ls=1000):
        """
        Coordinate descent with Newton directions for L1-regularized logistic regression
        :param X: n x p feature matrix
        :param y: vector of labels in {-1, +1}
        :param max_iter:
        :return:
        """
        self._L = L
        y_ = y.copy()
        y_[y_ == 0] = -1
        n_items, n_features = X.shape
        if sparse.issparse(X):
            yX = X.multiply(y_.reshape((n_items, 1))).tocsc()
        else:
            yX = y_.reshape((n_items, 1)) * X
        if init_w is None:
            self._w = np.zeros(n_features)
        else:
            self._w = init_w
        self._v = np.zeros(n_features)
        self._active = np.ones(n_features, dtype=int)
        self._M = 0.0
        self._Q = self._D * self._compute_Q(self._w)
        self._R = self._C * np.sum(np.abs(self._w))
        self._expits = 1.0 / (1.0 + np.exp(-yX.dot(self._w)))
        self._exp_nyXw = np.exp(-yX.dot(self._w))
        self._f_val = self._compute_f(self._exp_nyXw, self._w)
        self._g = self._compute_gradients(yX)

        for k in range(max_epochs):
            delta, ls_steps = self._update(yX, k, randomize=randomize, max_ls=1000)
            w_sum = np.sum(np.abs(self._w))
            q_pen = self._compute_Q(self._w)
            if w_sum > 0:
                rel_change = delta / w_sum
            else:
                rel_change = 0.0
            if verbose > 1:
                print("epoch %d, delta=%0.5f, rel_change=%0.5f, ls_steps=%d, q_pen=%.3f" % (k, delta, rel_change, ls_steps, q_pen))
            if rel_change < tol and k >= min_epochs - 1:
                if verbose > 0:
                    print("relative change below tolerance; stoppping after %d epochs" % k)
                return

        if verbose > 0:
            print("Maximum epochs exceeded; stopping after %d epochs" % k)

    def pred_probs(self, X):
        n, p = X.shape
        probs = np.zeros([n, 2])
        prob_pos = expit(X.dot(self._w))
        probs[:, 1] = prob_pos
        probs[:, 0] = 1.0 - prob_pos
        return probs

    def get_params(self, deep=True):
        return {"C": self._C, "beta": self._beta, "sigma": self._sigma, "lower": self._lower,
                "upper": self._upper, "do_elimination": self._do_elimination}

    @profile
    def _update(self, yX, k, randomize=False, max_ls=1000):
        n_items, n_features = yX.shape
        running_abs_change = 0
        running_ls_steps = 0
        order = np.arange(n_features)
        if randomize:
            np.random.shuffle(order)
        for j in order:
            if self._active[j]:
                change, ls_steps = self._update_one_coordinate(yX, j, max_ls=1000)
                running_abs_change += np.abs(change)
                running_ls_steps += ls_steps
        # update the threshold for eliminating variables on the next iteration
        if self._do_elimination and k > 0:
            self._M = np.max(self._v / k)

        return running_abs_change, running_ls_steps

    @profile
    def _update_one_coordinate(self, yX, j, max_ls=1000):
        n_items, n_features = yX.shape

        #h = self._compute_hessian_element(yX[:, j])

        if sparse.issparse(yX):
            yX_j = np.array(yX[:, j].todense()).reshape((n_items, ))
        else:
            yX_j = yX[:, j]

        g = self._compute_grad_j(yX_j, j)
        h = self._compute_hessian_element(yX_j, j)
        if self._do_elimination:
            # start testing for feature elimination on the second epoch
            if self._M > 0:
                # if w is 0 and the gradient is small, eliminate the variable from the active set
                if self._w[j] == 0 and -1 + self._M < g < 1 - self._M:
                    #print("Eliminating %d" % j)
                    self._active[j] = 0
                    return 0, 0

            # compute a new value for updating self._M
            if self._w[j] > 0:
                self._v[j] = np.abs(g + 1)
            elif self._w[j] < 0:
                self._v[j] = np.abs(g - 1)
            else:
                self._v[j] = np.max([g - 1, -1 - g, 0])

        # do soft-thresholding
        if g + 1.0 <= h * self._w[j]:
            d = -(g + 1.0) / h
        elif g - 1.0 >= h * self._w[j]:
            d = -(g - 1.0) / h
        else:
            d = -self._w[j]

        # create a scaling factor for the step size
        a = 1.0  # my name for lambda

        # check upper and lower limits, and set max step accordingly
        if self._lower is not None:
            if self._w[j] + d < self._lower:
                diff = self._lower - self._w[j]
                a = diff / float(d)
        if self._upper is not None:
            if self._w[j] + d > self._upper:
                diff = self._upper - self._w[j]
                a = diff / float(d)

        # unless we've hit a bound, use line search to find how far to move in this direction
        i = 0
        if a > 0:
            # set up the threshold for convergence
            thresh = self._sigma * (g * d + self._C * np.abs(self._w[j] + d) - self._C * np.abs(self._w[j]))
            # remove the current weight from the stored quadratic form
            w_c = self.get_w()
            w_c[j] = 0
            Q_minus_w_j = self._Q - self._D * self._compute_Q(w_c)
            # remove the current weight from the stored 1-norm of weights
            R_minus_w_j = self._R - self._C * np.abs(self._w[j])
            # do line search
            f_new, a, i, exp_nyXw = self._line_search(yX_j, d, self._w[j], Q_minus_w_j, R_minus_w_j, a, thresh, j, max_ls=1000)
            # store the updated values
            self._f_val = f_new
            self._w[j] += a * d
            # add the 1-norm of the new weight back into the stored sum
            self._R = R_minus_w_j + self._C * np.abs(self._w[j])
            w_c[j] = self._w[j]
            self._Q = Q_minus_w_j + self._D * self._compute_Q(w_c)
            self._exp_nyXw = exp_nyXw
            # recompute the stored probabilities and gradient
            self._expits = self._compute_probs(yX)
            #self._g = self._compute_gradients(yX)

        return a * d, i

    @profile
    def _line_search(self, yX_j, d, prev_w_j, base_Q, base_R, a, thresh, j, max_ls=1000):
        i = 0
        n_items = len(self._exp_nyXw)
        step = a * d
        w_j = prev_w_j + step

        if sparse.issparse(yX_j):
            # This is really slow; leaving it here in case I want to cythonize it, but otherwise
            # it is better to just pass in a dense vector
            if not sparse.isspmatrix_coo(yX_j):
                yX_j = yX_j.tocoo()
            exp_nyXw = self._exp_nyXw.copy()
            for (row, value) in zip(yX_j.row, yX_j.data):
                exp_nyXw[row] *= np.exp(-step * value)
        else:
            exp_nyXw = self._exp_nyXw * np.exp(-step * yX_j)

        w_c = np.zeros(len(self._w))
        w_c[j] = w_j
        L = self._compute_L(exp_nyXw)
        Q = base_Q + self._D * self._compute_Q(w_c)
        R = base_R + self._C * np.abs(w_j)
        f_new = L + Q + R
        while f_new - self._f_val > a * thresh and i < max_ls:
            a *= self._beta
            step = a * d
            w_j = prev_w_j + step

            if sparse.issparse(yX_j):
                exp_nyXw = self._exp_nyXw.copy()
                for (row, value) in zip(yX_j.row, yX_j.data):
                    exp_nyXw[row] *= np.exp(-step * value)
            else:
                exp_nyXw = self._exp_nyXw * np.exp(-step * yX_j)

            L = self._compute_L(exp_nyXw)
            w_c[j] = w_j
            Q = base_Q + self._D * self._compute_Q(w_c)
            R = base_R + self._C * np.abs(w_j)
            f_new = L + Q + R
            i += 1
        return f_new, a, i, exp_nyXw

    @profile
    def _compute_probs(self, yX):
        return 1.0 / (1.0 + np.exp(-yX.dot(self._w)))

    @profile
    def _compute_f(self, exp_nyXw, w):
        return self._compute_L(exp_nyXw) + self._D * self._compute_Q(w) + self._C * self._compute_R(w)

    @profile
    def _compute_L(self, exp_nyXw):
        return np.sum(np.log(1.0 + exp_nyXw))

    @profile
    def _compute_R(self, w):
        return self._C * np.sum(np.abs(w))

    @profile
    def _compute_Q(self, w):
        return w.T @ self._L @ w

    @profile
    def _compute_gradients(self, yX):
        return (yX.T.dot(self._expits - 1.0)) + self._D * (2 * (self._L @ self._w))

    @profile
    def _compute_grad_j(self, yX_j, j):
        grad_b = 2 * (self._L[j] @ self._w)
        return (yX_j.dot(self._expits - 1.0)) + self._D * grad_b

    @profile
    def _compute_hessian_element(self, yX_j, j):
        if sparse.issparse(yX_j):
            # This is kind of slow, unless very sparse; probably better to pass in a dense vector
            return yX_j.T.power(2).multiply(self._expits * (1.0 - self._expits)).sum()
        else:
            return (np.sum(yX_j ** 2 * self._expits * (1.0 - self._expits))) + self._D * 2 * self._L[j, j]

    def get_w(self):
        return self._w.copy()


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--elim', action="store_true", dest="elimination", default=False,
                      help='Do heuristic variable elimination: default=%default')
    parser.add_option('-n', dest='n', default=1000,
                      help='Number of instances: default=%default')
    parser.add_option('-p', dest='p', default=50,
                      help='Number of features: default=%default')
    parser.add_option('-s', dest='sparsity', default=0.5,
                      help='Expected proportion of zero entries in X: default=%default')
    parser.add_option('--sparse', action="store_true", dest="sparse", default=False,
                      help='Cast data to a scipy.sparse matrix: default=%default')
    parser.add_option('--nonlinear', action="store_true", dest="nonlinear", default=False,
                      help='Generate nonlinear data for testing: default=%default')
    parser.add_option('--seed', dest='seed', default=None,
                      help='Random seed: default=%default')
    parser.add_option('--skl', action="store_true", dest="skl", default=False,
                      help='Use sklearn implementation: default=%default')
    parser.add_option('-v', dest='verbose', default=0,
                      help='Verbosity level: default=%default')

    (options, args) = parser.parse_args()

    do_elimination = options.elimination
    n = int(options.n)
    p = int(options.p)
    sparsity = float(options.sparsity)
    do_sparse = options.sparse
    nonlinear = options.nonlinear
    seed = options.seed
    use_skl = options.skl
    verbose = int(options.verbose)

    if seed is not None:
        np.random.seed(int(seed))

    #X = np.array(np.random.randint(low=0, high=2, size=(n, p)), dtype=np.float64)
    X = np.array(np.random.binomial(p=1-sparsity, n=1, size=(n, p)), dtype=np.float64)
    beta = np.array(np.random.randn(p), dtype=np.float64) * np.random.randint(low=0, high=2, size=p)
    if verbose > 0:
        print(beta)

    # make a non-linear problem to encourage line search
    if nonlinear:
        X2 = X**2
        beta2 = np.array(np.random.randn(p), dtype=np.float64) * np.random.randint(low=0, high=2, size=p)
        ps = expit(np.dot(X, beta) + np.dot(X2, beta2))
    else:
        ps = expit(np.dot(X, beta))
    y = np.random.binomial(p=ps, n=1, size=n)

    if do_sparse:
        X = sparse.csc_matrix(X)

    if use_skl:
        model = LogisticRegression(C=1.0, penalty='l1', fit_intercept=False)
        model.fit(X, y)
        if verbose > 0:
            print(model.coef_)
        pred = model.predict(X)
        if verbose > 0:
            print(np.sum(np.abs(y - pred)) / float(n))

    else:
        y2 = y.copy()
        y2[y == 0] = -1

        solver = CDN(C=1.0, do_elimination=do_elimination)
        solver.fit(X, y2, max_epochs=200, randomize=True, verbose=verbose)
        if verbose > 0:
            print(solver.get_w())

        pred_probs = solver.pred_probs(X)
        pred = np.argmax(pred_probs, axis=1)
        if verbose > 0:
            print(np.sum(np.abs(y - pred)) / float(n))


if __name__ == '__main__':
    main()

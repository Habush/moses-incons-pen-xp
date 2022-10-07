__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from notebooks.variable_selection.util import *


class MahalaKernelReg:

    def __init__(self, gamma=1, K=None, cov=None):
        self.gamma = gamma
        self.K = K
        self.cov_ = cov

    def fit(self, X, y, G=None, alpha=1):
        self.graph_ = G
        self.krr_ = KernelRidge(kernel="precomputed", alpha=alpha)
        if self.cov_ is None and self.K is None:
            self.cov_ = get_emp_covariance(X, G)
        if self.K is None:
            self.K = calculate_mahal_kernel(X, X, self.cov_, self.gamma)

        self.krr_.fit(self.K, y)
        self.X_fitted_ = X
        return self

    def predict(self, X=None):
        if self.krr_ is None:
            raise RuntimeError("Estimator not fitted, fit the estimator first by running the fit method")
        if X is None: #predict on train
            return self.krr_.predict(self.K)

        K_x = calculate_mahal_kernel(X, self.X_fitted_, self.cov_, self.gamma)

        return self.krr_.predict(K_x)


def krr_gamma_opt(gammas, X, y, G, folds=5, idty=False):
    train_errs = np.zeros(len(gammas))
    val_errs = np.zeros(len(gammas))

    for k, g in enumerate(gammas):
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        train_f_errs = np.zeros(folds)
        test_f_errs = np.zeros(folds)
        print(f"---- Running {k} - gamma - {g} ----")
        i = 0
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            if idty:
                K = np.identity(X_train.shape[0])
                cov = np.identity(X_train.shape[1])
                mkrr = MahalaKernelReg(gamma=g, K=K, cov=cov) # to calculate the distance b/n test and train samples we need a covariance matrix - which is set an identity matrix
            else:
                mkrr = MahalaKernelReg(gamma=g)
            mkrr.fit(X_train, y_train, G=G)
            y_pred_train = mkrr.predict()
            y_pred_test = mkrr.predict(X_test)

            train_f_errs[i] = mean_squared_error(y_train, y_pred_train)
            test_f_errs[i] = mean_squared_error(y_test, y_pred_test)
            i += 1
        train_errs[k] = np.mean(train_f_errs)
        val_errs[k] = np.mean(test_f_errs)

    print("--- Done --- ")
    return train_errs, val_errs
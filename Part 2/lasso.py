import numpy as np


class LassoModel(object):

    def __init__(self):
        self.beta_ = None


    def fit(self, X, y, lambda_param=1.0, max_iter=200):
        assert(X.shape[0] == y.shape[0])
        self.beta_ = np.zeros(X.shape[1])

        for n_iter in range(max_iter):
            for j in range(X.shape[1]):
                tmp = self.beta_.copy()
                tmp[j] = 0

                # calculate residuals
                r_j = y - np.dot(X, tmp)
                beta_j_star = 1.0 / len(y) * np.dot(X[:, j], r_j)

                self.beta_[j] = self.soft_thresholding_(beta_j_star, lambda_param)
    

    def predict(self, X):
        y = np.dot(X, self.beta_)
        return y
    

    def soft_thresholding_(self, beta_j, lambda_param):
        if beta_j <= 0:
            return -max(-beta_j - lambda_param, 0.0)
        else:
            return max(beta_j - lambda_param, 0.0)
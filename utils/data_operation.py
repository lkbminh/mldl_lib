import numpy as np

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def RMSE(y_true, y_pred):
    return np.sqrt(MSE(y_true, y_pred))

def R2(y_true, y_pred):
    mean_ = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - mean_) ** 2)

    if ss_tot == 0:
        return 0
    
    return 1 - ss_res/ss_tot

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.maximum(np.std(X, axis=0), 1e-8)
        return self

    def transform(self, X):
        if self.mean_ == None:
            raise RuntimeError('fit has not been run')
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        if self.mean_ == None:
            raise RuntimeError('fit has not been run')
        return X * self.std_ + self.mean_
    
class Normalizer:
    def __init__(self, norm='l2'):
        self.norm = norm
        self.method_name = f"_{norm}_normalize"

    def fit(self, X):
        return self

    def transform(self, X):
        if hasattr(self, self.method_name):
            method = getattr(self, self.method_name)
            return method(X)
        else:
            raise ValueError(f"Unknown norm: {self.norm}")

    def _l1_normalize(self, X):
        norms = np.sum(np.abs(X), axis=1, keepdims=True)
        return X / np.maximum(norms, 1e-8)

    def _l2_normalize(self, X):
        norms = np.sqrt(np.sum(np.square(X), axis=1, keepdims=True))
        return X / np.maximum(norms, 1e-8)

    def _max_normalize(self, X):
        norms = np.max(np.abs(X), axis=1, keepdims=True)
        return X / np.maximum(norms, 1e-8)
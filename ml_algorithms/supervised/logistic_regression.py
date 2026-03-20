import numpy as np
from utils.data_manipulation import gen_batches, shuffle_data

class LogisticRegression:
    def __init__(self, learning_rate = 0.01, epochs = 100, mode_name = '_BGD'):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.parameters = None
        self.mode_name = mode_name

    def logf(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        X_temp = np.insert(X, 0, 1, axis=1)

        if hasattr(self, self.mode_name):
            mode = getattr(self, self.mode_name)
            mode(X_temp, y)
        else:
            raise RuntimeError(f'{self.mode_name} is undefined.')

    def predict_prob(self, X):
        X_temp = np.c_[np.ones(X.shape[0]), X]
        return self.logf(X_temp @ self.parameters)

    def predict(self, X, threshold=0.5):
        prob = self.predict_prob(X)
        return (prob >= threshold).astype(int)

    def _BGD(self, X, y):
        examples, features = X.shape
        self.parameters = np.zeros(features)

        for _ in range(self.epochs):
            h = self.logf(X @ self.parameters)
            gradient = (1/examples) * (X.T @ (h - y))
            self.parameters -= self.learning_rate * gradient

    def _SGD(self, X, y):
        examples, features = X.shape
        self.parameters = np.zeros(features)

        for epoch in range(self.epochs):
            X_shuffled, y_shuffled = shuffle_data(X, y)
            for _ in range(examples):
                X_sample, y_sample = X_shuffled[_], y_shuffled[_]

                h = self.logf(X_sample @ self.parameters)
                gradient = X_sample.T * (h - y_sample)
                self.parameters -= self.learning_rate * gradient

    def _MBGD(self, X, y, batch_size=64):   
        features = X.shape[1]
        self.parameters = np.zeros(features)

        for _ in range(self.epochs):
            batches = gen_batches(X, y, batch_size)

            for batch in batches:
                X_sample, y_sample = batch
                examples = X_sample.shape[0]

                h = self.logf(X_sample @ self.parameters)
                gradient = (1/examples) * (X_sample.T @ (h - y_sample))
                self.parameters -= self.learning_rate * gradient
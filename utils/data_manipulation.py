import numpy as np

def shuffle_data(X, y, random_state=None):
    if random_state:
        np.random.seed(random_state)
    idx = np.random.permutation(np.arange(X.shape[0]))
    return X[idx], y[idx]

def train_test_split(X, y, test_size=0.25, shuffle=True, random_state=None):
    if shuffle:
        X, y = shuffle_data(X, y, random_state)
    split_idx = int(X.shape[0] * (1 - test_size))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test

def gen_batches(X, y, batch_size=64, shuffle=True, random_state=None):
    if shuffle:
        X, y = shuffle_data(X, y, random_state)
    for i in range(0, X.shape[0], batch_size):
        yield X[i : i+batch_size], y[i : i+batch_size]

import numpy as np

def shuffle_data(X, y, random_state=None):
    if random_state:
        np.random.seed(random_state)
    idx = np.random.shuffle(np.arange(X.shope[0]))
    return X[idx], y[idx]

def train_test_split(X, y, test_size=0.75, shuffle=True, random_state=None):
    if shuffle:
        X, y = shuffle_data(X, y, random_state)
    size = np.floor(X.shape[0] * test_size)
    X_train, X_test = X[:size], X[size:]
    y_train, y_test = y[:size], y[size:]
    return X_train, X_test, y_train, y_test

def gen_batches(X, y, batch_size=64, shuffle=True, random_state=None):
    if shuffle:
        X, y = shuffle_data(X, y, random_state)
    for i in range(0, X.shape[0], batch_size):
        yield X[i : i+batch_size], y[i : i+batch_size]

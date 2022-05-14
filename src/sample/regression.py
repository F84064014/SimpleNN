from statistics import mode
from matplotlib import pyplot as plt
from _SimpleNN import FNN
from sklearn import datasets
import numpy as np

config = '''linear, 2, 1
MSE'''

if __name__ == "__main__":

    n = 1000

    def f(x):
        # return (0.64*(x**3) - 0.69*(x**2) - x + 69) / 60 + 4 * x - 800
        # return 2*x + 4
        return x**2 -2*x +3

    X = np.random.uniform(-50, 50, n)
    error = np.random.normal(0, 10, n)
    y = f(X) + error

    X = np.c_[X, np.square(X)]
    print(X.shape)

    random_index = np.arange(n)
    np.random.shuffle(random_index)

    split = int(n * 0.8)
    X_train, X_valid = X[random_index[:split]], X[random_index[split:]]
    y_train, y_valid = y[random_index[:split]], y[random_index[split:]]

    X_train = X_train.astype(np.float64)
    X_valid = X_valid.astype(np.float64)
    y_train = y_train.astype(np.float64)
    y_valid = y_valid.astype(np.float64)

    model = FNN(config)
    model.summary()

    model.fit(X_train, y_train, X_valid, y_valid, epochs=500, lr=0.00000025)
    #model.fit(X_train, y_train, epochs=3, lr=0.1)

    lins = np.c_[np.arange(-500, 500)/10, (np.arange(-500, 500)/10)**2]
    y_hat = model.evaluate(lins)

    plt.plot(X[:, 0], y, '.')
    plt.plot(np.arange(-500, 500)/10, y_hat)
    plt.show()

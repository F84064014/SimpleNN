from _SimpleNN import FNN
from sklearn import datasets
import numpy as np

config = '''linear, 10, 128, relu
linear, 128, 256, relu
linear, 256, 1
MSE'''

if __name__ == "__main__":


    diabetes = datasets.load_diabetes()

    X = diabetes.data
    y = diabetes.target

    n = len(X)
    random_index = np.arange(n)
    np.random.shuffle(random_index)

    split = int(n * 0.8)
    X_train, X_valid = X[random_index[:split]], X[random_index[split:]]
    y_train, y_valid = y[random_index[:split]], y[random_index[split:]]

    X_train = X_train.astype(np.float64)
    X_valid = X_valid.astype(np.float64)
    y_train = y_train.astype(np.float64)
    y_valid = y_valid.astype(np.float64)

    print(X.shape)

    model = FNN(config)
    model.summary()

    model.fit(X_train, y_train, X_valid, y_valid, epochs=500, lr=0.000005)
    #model.fit(X_train, y_train, epochs=3, lr=0.1)

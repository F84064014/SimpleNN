from SimpleNN import FNN
from sklearn import datasets
import numpy as np

config = '''linear, 4, 128, relu
linear, 128, 256, relu
linear, 256, 3, sigmoid
CrossEntropy'''

if __name__ == "__main__":


	iris = datasets.load_iris()
	
	X = iris.data
	y = iris.target

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

	model = FNN(config)
	model.summary()
	
	model.fit(X_train, y_train, X_valid, y_valid, epochs=300, lr=0.1)

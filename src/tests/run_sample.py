import _SimpleNN as snn
import numpy as np

from sklearn import datasets

config='''linear 2, 10, relu
linear 10, 2, sigmoid
CrossEntropy
'''

if __name__ == "__main__":

	model = snn.FNN(config)
	model.summary()

	iris = datasets.load_iris()
	X = iris.data[:, :2] # we only take first two features
	y = iris.target

import os
import cmath
import pytest
import timeit
import unittest
import numpy as np

import _SimpleNN as snn

class TestInstance(unittest.TestCase):

	def test_model(self):

		with open('./config/config.txt') as f:
			config_text = f.read()

		snn.FNN(config_text)


if __name__ == "__main__":

	X_train = np.ones((5, 5));
	X_valid = np.ones((3, 5));
	
	for i in range(X_train.shape[0]):
		for j in range(X_train.shape[1]):
			X_train[i, j] = i+j

	with open('./config/config.txt') as f:
		config_text = f.read()
	model = snn.FNN(config_text)
	model.summary()
	#model.fit(X_train, X_valid);
	out = model.evaluate(X_train)
	print(out)

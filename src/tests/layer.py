import os
import cmath
import pytest
import timeit
import unittest
import numpy as np

import _Layer

class TestInstance(unittest.TestCase):

	def test_dense(self):
		x = np.random.random((100, 10))
		dense = _Layer.Dense(10,10,"sigmoid")
		weight = dense.getw()
		bias = dense.getb()
		self.assertEqual(weight.shape, (10,10))

		x_dense = dense.forward(x)
		x = x.dot(weight) + bias
		x = 1 / (1 + np.exp(-x))

		for i in range(100):
			for j in range(10):
				self.assertAlmostEqual(x[i, j], x_dense[i, j])


	def test_relu(self):
		x = np.random.random((100, 10))
		relu = _Layer.ReLu()
		
		x_relu = relu.forward(x)
		x = np.where(x>0, x, 0)
		
		for i in range(100):
			for j in range(10):
				self.assertAlmostEqual(x[i, j], x_relu[i, j])

	def test_sigmoid(self):
		x = np.random.random((100, 10))
		sigmoid = _Layer.Sigmoid()
		
		x_sigmoid = x.copy()
		sigmoid.forward(x_sigmoid)
		x = 1/(1+np.exp(-x))

		for i in range(100):
			for j in range(10):
				self.assertAlmostEqual(x[i, j], x_sigmoid[i, j])


if __name__ == "__main__":

	x = np.random.random((100, 10))
	dense = _Layer.Dense(10,10,"sigmoid")
	weight = dense.getw()
	print(weight.dtype)
	dense.forward(x)
	#print(weight)

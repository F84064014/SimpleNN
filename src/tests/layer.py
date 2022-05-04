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
		x_input = x.copy()
		dense = _Layer.Dense(10,10,"sigmoid")
		weight = dense.getw()
		bias = dense.getb()
		self.assertEqual(weight.shape, (10,10))

		x_dense = dense.forward(x)
		x = x.dot(weight) + bias
		out = x.copy()
		x = 1 / (1 + np.exp(-x))

		# test forward
		for i in range(100):
			for j in range(10):
				self.assertAlmostEqual(x[i, j], x_dense[i, j])

		grad_out = np.zeros((100,10))
		grad_dense = dense.backward(x_input, grad_out)
		grad = grad_out * (np.exp(-out))/((np.exp(-out)+1)**2)
		grad = grad.dot(weight.T)

		# test backward
		self.assertEqual(grad.shape[0], grad_dense.shape[0])
		self.assertEqual(grad.shape[1], grad_dense.shape[1])
		for i in range(grad.shape[0]):
			for j in range(grad.shape[1]):
				self.assertAlmostEqual(grad_dense[i,j], grad[i,j])

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

	# x = np.random.random((100, 10))
	# dense = _Layer.Dense(10,10,"sigmoid")
	# weight = dense.getw()
	# print(weight.dtype)
	# dense.forward(x)
	x = np.random.random((100, 10))
	x_input = x.copy()
	dense = _Layer.Dense(10,10,"sigmoid")
	weight = dense.getw()
	bias = dense.getb()

	x_dense = dense.forward(x)
	x = x.dot(weight) + bias
	out = x.copy()
	x = 1 / (1 + np.exp(-x))

	grad_out = np.random.random((100,10))
	grad_dense = dense.backward(x_input, grad_out)
	'''
	grad = grad_out * (np.exp(-out))/((np.exp(-out)+1)**2)
	grad = grad_out * weight.T

	# test backward
	self.assertEqual(grad.shape[0], grad_dense.shape[0])
	self.assertEqual(grad.shape[1], grad_dense.shape[1])
	for i in range(grad.shape[0]):
		for j in range(grad.shape[1]):
			self.assertAlmostEqual(grad_dense[i,j], grad)
	'''
	#print(weight)

import os
import cmath
import json
import pytest
import timeit
import unittest
import numpy as np

import _testNN

def sigmoid_forward(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_back(x):
	return (np.exp(-x))/((np.exp(-x)+1)**2)

def relu_back(x):
	return np.where(x>0, 1, 0)

class TestInstance(unittest.TestCase):

	# assume input 1000 * 128
	# assume weight 128 * 256, bias 1000 * 256
	def active_back(self, dactivation, activation):

		n = 1000
		in_dim = 128
		out_dim = 256

		x_input = np.random.uniform(-10, 10, (n, in_dim))
		grad_output = np.random.uniform(-10, 10, (n, out_dim))

		m_weight = np.random.uniform(-10, 10, (in_dim, out_dim))
		m_bias = np.random.uniform(-10, 10, out_dim)
		m_weight_test = m_weight.copy()
		m_bias_test = m_bias.copy()
		m_out = np.random.uniform(-10, 10, grad_output.shape)

		temp = grad_output * dactivation(m_out)
		grad_input = np.dot(temp, m_weight.T)

		grad_weights = np.dot(x_input.T, temp)
		grad_biases = temp.mean(axis=0)*x_input.shape[0]

		#m_out = 0

		grad_input_test = _testNN.backward(x_input, grad_output, m_weight_test, m_bias_test, m_out, activation)
	
		for i in range(m_weight.shape[0]):
			for j in range(m_weight.shape[1]):
				self.assertNotEqual(m_weight[i, j], m_weight_test[i, j])

		m_weight = m_weight - 0.1 * grad_weights
		m_bias = m_bias - 0.1 * grad_biases

		for i in range(grad_input.shape[0]):
			for j in range(grad_input.shape[1]):
				self.assertAlmostEqual(grad_input[i, j], grad_input_test[i, j])
		
		for i in range(m_weight.shape[0]):
			for j in range(m_weight.shape[1]):
				self.assertAlmostEqual(m_weight[i, j], m_weight_test[i, j])

		for i in range(m_bias.shape[0]):
				self.assertAlmostEqual(m_bias[i], m_bias_test[i])

	def test_back_sigmoid(self):
		self.active_back(sigmoid_back, "Sigmoid")

	def test_back_relu(self):
		self.active_back(relu_back, "ReLu")

	def test_softmax(self):
		
		n, p = 1000, 20
		m1 = np.random.uniform(-1, 1, (n, p))
		m1_copy = m1.copy()
		m2 = np.exp(m1) / np.exp(m1).sum(axis=-1, keepdims=True)
		m3 = _testNN.cal_softmax(m1)

		for i in range(m2.shape[0]):
			for j in range(m2.shape[1]):
				self.assertAlmostEqual(m2[i, j], m3[i, j])
		
		for i in range(m1.shape[0]):
			for j in range(m1.shape[1]):
				self.assertAlmostEqual(m1[i, j], m1_copy[i, j])

	def test_cal_grad(self):
		
		n, p = 1000, 20
		y_out = np.random.uniform(-1, 1, (n, p))
		y_true = np.random.randint(0, p, n)

		softmax = np.exp(y_out) / np.exp(y_out).sum(axis=-1, keepdims=True)
		onehot = np.zeros_like(y_out)
		onehot[np.arange(n), y_true] = 1

		grad = (softmax - onehot) / n
		grad_test = _testNN.cal_grad(y_out, y_true.reshape(n, 1).astype(np.float64))

		for i in range(grad.shape[0]):
			for j in range(grad.shape[1]):
				self.assertAlmostEqual(grad[i,j], grad_test[i,j])

	def test_forward(self):

		n, p, q = 1000, 20, 128
		data = np.random.uniform(-10, 10, (n,p))
		w = np.random.uniform(-1, 1, (p, q))
		b = np.random.uniform(-1, 1, q)
		m_out = np.zeros((n, q))

		y = data.dot(w) + b
		y_a = sigmoid_forward(y)
		y_test = _testNN.forward(data, w, b, m_out)

		for i in range(len(y)):
			self.assertAlmostEqual(y_a[i, 0], y_test[i, 0])

		for i in range(len(y)):
			self.assertAlmostEqual(y[i, 0], m_out[i, 0])


	def test_bitwise(self):

		m1 = np.random.uniform(-10, 10, (300, 500))
		m2 = np.random.uniform(-10, 10, (300, 500))
		
		m3 = m1 * m2
		m4 = _testNN.bitwise_mul(m1, m2)

		for i in range(m1.shape[0]):
			for j in range(m1.shape[1]):
				self.assertAlmostEqual(m3[i, j], m4[i, j])


	def test_dsigmoid(self):

		m1 = np.random.uniform(-1, 1, (500,500))

		m2 = sigmoid_back(m1)
		m3 = _testNN.sigmoid_backward(m1)

		for i in range(m1.shape[0]):
			for j in range(m2.shape[1]):
				self.assertAlmostEqual(m2[i,j], m3[i,j])

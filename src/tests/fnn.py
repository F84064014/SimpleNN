import os
import cmath
import json
import pytest
import timeit
import unittest
import numpy as np

import _SimpleNN


config = '''linear, 4, 4, relu
linear, 4, 2, sigmoid
CrossEntropy
'''

config2 = '''linear, 10, 10, relu
linear, 10, 4, sigmoid
CrossEntropy
'''

class TestInstance(unittest.TestCase):

	def load_weight(self, fp='w.txt'):

		with open(fp, 'r') as f:
			lines = f.readlines()

		d = json.loads(lines[0])	
		layer1 = d['w']
		b1 = np.array(d['b'])
		w1 = np.zeros((4, 4))
		for i in range(4):
			for j in range(4):
				w1[i, j] = layer1[i*4+j]


		d = json.loads(lines[1])
		layer2 = d['w']
		b2 = np.array(d['b'])
		w2 = np.zeros((4, 2))
		for i in range(4):
			for j in range(2):
				w2[i, j] = layer2[i*2+j]

		return w1, b1, w2, b2
					

	def test_evaluate(self):
		
		np.random.seed(42)
		test_shape = (4, 4)
		x = np.random.random(test_shape).astype(np.float64)
		y_true = np.random.randint(0, 2, size=4).astype(np.float64)

		model = _SimpleNN.FNN(config)
		model.save('./w.txt')
		y_model = model.evaluate(x)
	

		w1, b1, w2, b2 = self.load_weight()

		_x = x.copy()
		_x = _x.dot(w1) + b1
		_x = np.where(_x>0, _x, 0)
		_x = _x.dot(w2) + b2
		_x = 1 / (1+np.exp(-_x))
		
		for i in range(4):
			for j in range(2):
				# somehow get error if count to 7 place
				self.assertAlmostEqual(_x[i, j], y_model[i, j], 6)

	def test_loss(self):

		def softmax_crossentropy_with_logits(logits,reference_answers):
			
			# Compute crossentropy from logits[batch,n_classes] and ids of correct answers
			logits_for_answers = logits[np.arange(len(logits)),reference_answers.astype(np.int32)]

			xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))

			return xentropy

		model = _SimpleNN.FNN(config)

		# n: number of data
		# p: number of demension
		n, p = 10, 4

		y = np.random.random(size=(n, p))
		y_true = np.random.randint(0, p, n).astype(np.float64)

		loss_model = model.getLoss(y, y_true)
		loss = softmax_crossentropy_with_logits(y, y_true).sum()

		self.assertAlmostEqual(loss, loss_model, 6)

	def test_loss_grad(self):
		
		def grad_softmax_crossentropy_with_logits(logits,reference_answers):
			
			# Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers
			ones_for_answers = np.zeros_like(logits)
			ones_for_answers[np.arange(len(logits)),reference_answers] = 1

			softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)

			return (- ones_for_answers + softmax) / logits.shape[0]
	
		model = _SimpleNN.FNN(config)

		# n: number of data
		# p: number of demension
		n, p = 10, 4

		y = np.random.random(size=(n,p))
		y_true = np.random.randint(0, p, n).astype(np.float64)

		grad_model = model.getGrad(y, y_true)
		grad = grad_softmax_crossentropy_with_logits(y, y_true.astype(np.int32))

		for i in range(grad.shape[0]):
			for j in range(grad.shape[1]):
				self.assertAlmostEqual(grad[i, j], grad_model[i, j])


	def test_backward(self):
		
		def grad_softmax_crossentropy_with_logits(logits,reference_answers):
			
			# Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers
			ones_for_answers = np.zeros_like(logits)
			ones_for_answers[np.arange(len(logits)),reference_answers] = 1

			softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)

			return (- ones_for_answers + softmax) / logits.shape[0]

		def sigmoid_df_dx(x):
			return (np.exp(-x))/((np.exp(-x)+1)**2)


		model = _SimpleNN.FNN(config)
		model.save('./w_init.txt')
		w1_init, b1_init, w2_init, b2_init = self.load_weight('./w_init.txt')

		X_train = np.random.random(size=(100, 4))
		y_train = np.random.randint(0, 2, 100)

		y_out = model.evaluate(X_train)
		model.fit(X_train, y_train.astype(np.float64), lr=0.1, epochs=1)
		
		model.save('./w_update.txt')
		w1_update, b1_update, w2_update, b2_update = self.load_weight('./w_update.txt')

		grad_loss = grad_softmax_crossentropy_with_logits(y_out, y_train)

		X_inter = X_train.dot(w1_init)+b1_init
		grad_output = grad_loss * sigmoid_df_dx(X_inter.dot(w2_init)+b2_init)
		grad_weights = np.dot(X_inter.T, grad_output)

		w2_true = w2_init - grad_weights * 0.1

		for i in range(w2_update.shape[0]):
			for j in range(w2_update.shape[1]):
				self.assertAlmostEqual(w2_update[i, j], w2_true[i,j], 6)

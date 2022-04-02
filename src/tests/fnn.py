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

class TestInstance(unittest.TestCase):

	def load_weight(self):

		with open('w.txt', 'r') as f:
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

			print(np.sum(np.exp(logits), axis=-1))

			xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))

			return xentropy

		model = _SimpleNN.FNN(config)

		y = np.random.random(size=(10, 4))
		y_true = np.random.randint(0, 4, 10).astype(np.float64)

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

		y = np.random(random(size=(10,4))
		y_true = np.random.randint(0,4,10).astype(np.float64)

		loss_model = model.backward()

		

def softmax_crossentropy_with_logits(logits,reference_answers):
	# Compute crossentropy from logits[batch,n_classes] and ids of correct answers
	logits_for_answers = logits[np.arange(len(logits)),reference_answers.astype(np.int32)]
	print(logits_for_answers)

	xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))

	return xentropy

if __name__ == "__main__":

	#m = TestInstance()
	#w1, b1, w2, b2 = m.load_weight()
	model = _SimpleNN.FNN(config)
	#model.summary()
	#test_shape = (4, 4)
	#x = np.random.random(test_shape)

	#model.evaluate(x)
	
	y = np.random.random((10, 4))
	y_true = np.random.randint(0, 4, 10).astype(np.float64)

	loss_model = model.getLoss(y, y_true.astype(np.float64))

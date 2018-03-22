from __future__ import print_function
import numpy as np

#linux env for matplot
try:
	import matplotlib
	matplotlib.use('Agg')  
	import matplotlib.pyplot as plt
except ImportError:
	pass

import chainer
import chainer.functions as F
import chainer.links as L

from chainer import reporter
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer import serializers 
import cupy


class LM(chainer.Chain):

	def __init__(self, n_units, n_out, n_kernel):
		super(LM, self).__init__()
		with self.init_scope():
			self.l_out = L.Linear(None, n_out)

	def __call__(self, x):
		y = self.l_out(x)
		y = F.absolute(y)
		return y


# full connected , use relu for the activation
class MLP(chainer.Chain):
	def __init__(self, n_unit, n_kernel = 1):
		super(MLP, self).__init__()
		with self.init_scope():
			self.l1 = L.Linear(None, n_unit)
			self.l_out = L.Linear(None, 2)

	def __call__(self, x):
		h1 = F.relu(self.l1(x)) # using relu active function
		y =  self.l_out(h1)
		return y


# definition of classical CNN
class CNN(chainer.Chain):
	def __init__(self,  h_unit=100, n_kernel=1):
		super(CNN, self).__init__()
		with self.init_scope():
			self.conv1 = L.Convolution2D(n_kernel, 32, (1, 7))  # in_channel, out_kernel, filter-size, stride, pad_0
			self.conv2 = L.Convolution2D(32, 64, (1, 3))
			self.conv3 = L.Convolution2D(64, 128, (1, 3))
			
			self.fc = L.Linear(None, h_unit)
			self.lo = L.Linear(None,2)

	def __call__(self, x): 
		
		h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 3)
		h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), 3)
		hf = self.fc(h2)
		y = self.lo(hf)

		return y


class Augmentor(chainer.Chain):
	def __init__(self, predictor):
		super(Augmentor, self).__init__(predictor=predictor)

	def __call__(self, x, t):

		y = self.predictor(x)
		self.loss = F.softmax_cross_entropy(y, t)
		self.acc = F.accuracy(y, t)

		reporter.report({'LOSS': self.loss}, self)
		reporter.report({'ACC': self.acc}, self)

		return self.loss

	def predict(self, x):
		y = self.predictor(x)
		return y



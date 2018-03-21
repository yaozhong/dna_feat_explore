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
	def __init__(self, n_unit, n_out, n_kernel = 1):
		super(MLP, self).__init__()
		with self.init_scope():
			self.l1 = L.Linear(None, n_unit)
			self.l_out = L.Linear(None, n_out)

	def __call__(self, x):
		h1 = F.relu(self.l1(x)) # using relu active function
		y =  self.l_out(h1)
		y = F.absolute(y)
		return y



class CNN(chainer.Chain):
	def __init__(self, n_kernel=1):
		super(CNN, self).__init__()
		with self.init_scope():
			self.conv1 = L.Convolution2D(n_kernel, 32, (1, 7))  # in_channel, out_kernel, filter-size, stride, pad_0
			self.conv2 = L.Convolution2D(32, 64, (1, 3))
			self.conv3 = L.Convolution2D(64, 128, (1, 3))
		
			self.bn1 = L.BatchNormalization(size=32)
			self.bn2 = L.BatchNormalization(size=64)
			self.bn3 = L.BatchNormalization(size=128)
			
			self.lo = L.Linear(None,2)

	def __call__(self, x, t): 
		
		h1 = F.leaky_relu((self.conv1(x)))
		h2 = F.leaky_relu((self.conv2(h1)))
		#h3 = F.leaky_relu((self.conv3(h2)))
		#y = F.log_softmax(self.lo(h1))
		y = self.lo(h2)

		accuracy = F.accuracy(F.log_softmax(y), t)
		reporter.report({'ACC': accuracy}, self)

		return y


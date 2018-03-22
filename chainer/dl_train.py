from __future__ import division
from util import *
from dl_model import *
from chainer.datasets import tuple_dataset

import argparse
import numpy as np
import cupy as cp

"""
Implemented functions:
[o] 1. Loading GTF file, do the option selection of the target regions. 
[o] 2. Extract the sequences of the given region. 
3. Transfer sequences to k-mers and prepare the training format. 
"""

def genDLTrainData(seqLen=100, split=0.8, cache=True):

	seq_pair = getData(seqLen)
	print("\n@ Totally selecting sequences: Ex-seq:{}, NC-seq:{}".format(len(seq_pair["ex"]), len(seq_pair["nc"])))

	# prepare the training and testing data.
	ex_seqs = seq_pair["ex"]
	nc_seqs = seq_pair["nc"]

	X_ex = np.array([seq2onehot(x) for x in ex_seqs ],dtype=np.float32)
	X_nc = np.array([seq2onehot(x) for x in nc_seqs],dtype=np.float32)

	Y_ex = np.ones(len(ex_seqs), dtype=np.int32)	
	Y_nc = np.zeros(len(nc_seqs), dtype=np.int32)

	tidx =  int(split * len(ex_seqs))
	X_ex_part1, X_ex_part2 = np.split(X_ex, [tidx])
	Y_ex_part1, Y_ex_part2 = np.split(Y_ex, [tidx])

	tidx =  int(split * len(nc_seqs))
	X_nc_part1, X_nc_part2 = np.split(X_nc, [tidx])
	Y_nc_part1, Y_nc_part2 = np.split(Y_nc, [tidx])

	X_train = np.concatenate((X_ex_part1, X_nc_part1), 0)
	X_test = np.concatenate((X_ex_part2, X_nc_part2), 0)

	Y_train = np.concatenate((Y_ex_part1, Y_nc_part1), 0)
	Y_test = np.concatenate((Y_ex_part2, Y_nc_part2), 0)

	## reshape, note in the binary classification case Y should be reshaped in 1 dimension.
	train = tuple_dataset.TupleDataset(X_train.reshape(X_train.shape[0], X_train.shape[1], 1, X_train.shape[2]), Y_train)
	test = tuple_dataset.TupleDataset(X_test.reshape(X_test.shape[0],X_test.shape[1], 1, X_test.shape[2]), Y_test)

	# caching the data
	if cache == False:
		np.save("/trainX", X_train)
		np.save("/trainY", Y_train)
		np.save("/testX", X_test)
		np.save("/testY", Y_test)

	return train, test


def trainDL(seqLen=100):

	parser = argparse.ArgumentParser(description='coding and non-coding sequence distinguish')

	## training options
	parser.add_argument('--batchsize', '-mb', type=int, default=128, help='Number of bins in each mini-batch')
	parser.add_argument('--epoch', '-e', type=int, default=100, help='Number of iters over the dataset for train')
	parser.add_argument('--out', '-o', default='train_curve', help='Directory to output the result')
	parser.add_argument('--modelPath', '-mp', default='./model', help='model output path')

	parser.add_argument('--frequency', '-f', type=int, default=-1, help='Frequency of taking a snapshot')
	parser.add_argument('--resume', '-r', default='', help='Resume the raining from snapshot')

	parser.add_argument('--gpu', '-g', type=int, default=1, help='GPU ID (if none -1)')
	parser.add_argument('--binSize', '-b', type=int, default=100, help='binSize')

	parser.add_argument('--unit', '-u', type=int, default=64, help='Number of units')
	parser.add_argument('--model', '-m', default=CNN, help='deep learning model')
	
	args = parser.parse_args()
	n_kernel = 4

	train, test = genDLTrainData(seqLen)
	## now training deep learning models from here

	print '-----------Data Loading Done---------------'
	print('\n# [Model]:{}'.format(args.model))
	print('# [GPU]: {}'.format(args.gpu))
	print('# [unit]: {}'.format(args.unit))
	print('# [Minibatch-size]: {}'.format(args.batchsize))
	print('# [epoch]: {}'.format(args.epoch))
	print('')

	model = Augmentor(CNN(1000, n_kernel))

	if args.gpu >= 0:
		chainer.cuda.get_device_from_id(args.gpu).use()
		model.to_gpu()

	optimizer = chainer.optimizers.Adam()
	optimizer.setup(model)

	train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
	test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

	# setup a trainer
	updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
	trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
	trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

	trainer.extend(extensions.LogReport(log_name="train.log"))
	trainer.extend(extensions.dump_graph('main/ACC'))

	if extensions.PlotReport.available():
		trainer.extend(extensions.PlotReport(['main/ACC', 'validation/ACC'], 'epoch', file_name= "acc_curve.png"))

	trainer.extend(extensions.PrintReport(['epoch', 'main/ACC', 'main/LOSS', 'validation/main/ACC', 'validation/main/LOSS','elapsed_time']))
	trainer.extend(extensions.ProgressBar())

	## running the training process
	trainer.run()
	
	## saving the model 
	if not os.path.exists(args.modelPath):
		os.makedirs(args.modelPath)
	serializers.save_npz(args.modelPath +'/'+ outFileName +'.model', model)


if __name__ == "__main__":
	trainDL(100)




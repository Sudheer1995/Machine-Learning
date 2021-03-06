#!/usr/bin/env

from __future__ import division
import csv
import numpy
import sys 

def load_data(train_file, test_file):
	"""loads data from csv file"""
	with open(train_file) as f:
		reader = csv.reader(f, delimiter=' ', quotechar='|')
		data = list(reader)
		samples = len(data)
		train_X = numpy.zeros((samples, 784), dtype=numpy.int)
		train_Y = numpy.zeros((samples, 1), dtype=numpy.int)
		for i in range(samples):
			row = map(int, data[i][0].split(','))
			train_Y[i, 0] = row[0]
			train_X[i, :] = numpy.asarray(row[1: ])
		train_X = numpy.append(numpy.ones((train_X.shape[0], 1)), train_X, axis=1)

	with open(test_file) as f:
		reader = csv.reader(f, delimiter=' ', quotechar='|')
		data = list(reader)
		samples = len(data)
		test_X = numpy.zeros((samples, 783), dtype=int)
		for i in range(samples):
			row = map(int, data[i][0].split(','))
			test_X[i, :] = numpy.asarray(row[1: ])
		test_X = numpy.append(numpy.ones((test_X.shape[0], 1)), test_X, axis=1)
		
		return (train_X, train_Y, test_X)

def train_weights(data, labels, sm_margin, bm_margin):
	"""trains weights for four perceptrons"""
	epoch = 0
	labels = 2*labels - 1	
	data = numpy.multiply(labels, data)
	s_weights = numpy.ones((data.shape[1], 1), dtype=numpy.float32)
	sm_weights = numpy.ones((data.shape[1], 1), dtype=numpy.float32)
	b_weights = numpy.ones((data.shape[1], 1), dtype=numpy.float32)
	bm_weights = numpy.ones((data.shape[1], 1), dtype=numpy.float32)
	while epoch < 100:
		for i in range(data.shape[0]):
			s_error = numpy.matmul(data[i, :].reshape(1, data.shape[1]), s_weights)
			sm_error = numpy.matmul(data[i, :].reshape(1, data.shape[1]), sm_weights)
			if s_error[0, 0] < 0:
				s_weights = numpy.add(s_weights, data[i, :].reshape(s_weights.shape[0], 1))
			if sm_error[0, 0] < sm_margin:
				sm_weights = numpy.add(sm_weights, data[i, :].reshape(sm_weights.shape[0], 1))

		b_errors = numpy.matmul(data, b_weights)
		bm_errors = numpy.matmul(data, bm_weights)
		bm_wrong_vecs, j = numpy.where(bm_errors < bm_margin)
		b_wrong_vecs, j = numpy.where(b_errors < 0)
		for k in bm_wrong_vecs:
			misclassified = data[k, :].reshape(bm_weights.shape[0], 1)
			bm_weights = numpy.add(bm_weights, misclassified)

		for k in b_wrong_vecs:
			misclassified = data[k, :].reshape(b_weights.shape[0], 1)
			b_weights = numpy.add(b_weights, misclassified)
		epoch += 1

	return (s_weights, sm_weights, b_weights, bm_weights)

def test_weights(weights, data, margin):
	"""returns results on test data"""
	data = numpy.append(numpy.ones((data.shape[0], 1)), data, axis=1)
	errors = numpy.matmul(data, weights)
	for i in range(errors.shape[0]):
		if errors[i, 0] <= margin:
			print 0
		else:
			print 1
	
if __name__ == '__main__':

	train = sys.argv[1]
	test = sys.argv[2]
	margin = 2
	train_X, train_Y, test_X = load_data(train, test)

	s_weights, sm_weights, b_weights, bm_weights = train_weights(train_X, train_Y, margin, margin)
	s_accuracy = test_weights(s_weights, test_X, 0)
	sm_accuracy = test_weights(sm_weights, test_X, margin)
	b_accuracy = test_weights(b_weights, test_X, 0)
	bm_accuracy = test_weights(bm_weights, test_X, margin)
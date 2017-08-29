#!/usr/bin/env

from __future__ import division
from numpy import linalg
import numpy
import csv
import sys

def load_data(train_file, test_file):
	"""loads data from csv file"""
	with open(train_file) as f:
		reader = csv.reader(f, delimiter=' ', quotechar='|')
		data = list(reader)
		samples = len(data)
		train_X = numpy.zeros((samples, 9), dtype=numpy.float32)
		train_Y = numpy.zeros((samples, 1), dtype=numpy.float32)
		for i in range(samples):
			row = map(numpy.float32, data[i][0].split(','))
			train_Y[i, 0] = row[10]
			train_X[i, :] = numpy.asarray(row[1: 10])
		train_X = numpy.append(numpy.ones((train_X.shape[0], 1)), train_X, axis=1)
	
	with open(test_file) as f:
		reader = csv.reader(f, delimiter=' ', quotechar='|')
		data = list(reader)
		samples = len(data)
		test_X = numpy.zeros((samples, 9), dtype=numpy.float32)
		for i in range(samples):
			row = map(numpy.float32, data[i][0].split(','))
			test_X[i, :] = numpy.asarray(row[1: ])			
		test_X = numpy.append(numpy.ones((test_X.shape[0], 1)), test_X, axis=1)

		return train_X, train_Y, test_X

def train_weights(data, labels, margin, ep):
	"""trains weights for batch perceptron with relaxation & margin"""
	size = 0
	epoch = 0
	labels -= 3
	data = numpy.multiply(labels, data)
	weights = numpy.random.rand(data.shape[1], 1)
	sm_weights = numpy.zeros((ep, data.shape[1]))
	counts = numpy.zeros((ep, 1))
	while epoch <= ep:
		k = epoch % data.shape[0]
		error = numpy.matmul(data[k, :].reshape(1, data.shape[1]), weights)
		if error < margin:
			sm_weights[size, :] = weights.reshape(1, data.shape[1])
			Z = data[k, :] * (margin - error[0, 0])/(linalg.norm(data[k, :])**2)
			weights = numpy.add(weights, Z.reshape(data.shape[1], 1))
			size += 1
		else:
			counts[size, 0] += 1
		epoch += 1

	return weights, sm_weights[:size, :], counts[:size, :]

def test_weights(weights, data, margin):
	"""returns results on test data"""
	errors = numpy.matmul(data, weights)
	for i in range(errors.shape[0]):
		if errors[i, 0] < margin:
			print 2
		else:
			print 4

def test_modified_weights(weights, counts, data):
	"""returns results on test data"""
	for i in range(data.shape[0]):
		temp = data[i, :].reshape(1, data.shape[1])*weights
		errors = numpy.sum(temp, axis=1)
		if numpy.sum(counts*numpy.sign(errors)) < 0:
			print 2
		else:
			print 4
			

if __name__ == '__main__':
	
	train = sys.argv[1]
	test = sys.argv[2]
	
	ep = 100000
	margin = 1
	train_X, train_Y, test_X = load_data(train, test)
	s_weights, sm_weights, counts = train_weights(train_X, train_Y, margin, ep)
	test_weights(s_weights, test_X, margin)
	test_modified_weights(sm_weights, counts, test_X)

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
	epoch = 0
	labels -= 3
	data = numpy.multiply(labels, data)
	prev_i = range(data.shape[0])
	weights = numpy.random.rand(data.shape[1], 1)
	while epoch <= ep:
		errors = numpy.matmul(data, weights)
		i, j = numpy.where(errors < margin)
		if len(prev_i) > len(i):
			for k in i:
				misclassified = data[k, :].reshape(weights.shape[0], 1)
				error = numpy.matmul(numpy.transpose(weights), misclassified)
				Z = misclassified * (margin - error[0, 0])/(linalg.norm(misclassified)**2)
				weights = numpy.add(weights, Z)
			prev_i = i
		epoch += 1

	return weights, (1-len(prev_i)/data.shape[0])*100

def test_weights(weights, data, margin):
	"""returns results on test data"""
	errors = numpy.matmul(data, weights)
	for i in range(errors.shape[0]):
		if errors[i, 0] < margin:
			print 2
		else:
			print 4

if __name__ == '__main__':
	
	train = sys.argv[1]
	test = sys.argv[2]
	
	ep = 1000
	margin = 3
	train_X, train_Y, test_X = load_data(train, test)
	weights, train_accuracy = train_weights(train_X, train_Y, margin, ep)
	test_accuracy = test_weights(weights, test_X, margin)
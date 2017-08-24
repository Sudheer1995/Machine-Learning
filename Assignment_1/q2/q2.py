#!/usr/bin/env

from __future__ import division
from numpy import linalg
import numpy
import csv
import sys
import time

def load_data(train_file):

	with open(train_file) as f:

		reader = csv.reader(f, delimiter=' ', quotechar='|')
		data = list(reader)
		samples = len(data)

		train_X = numpy.zeros((samples, 8), dtype=numpy.float32)
		train_Y = numpy.zeros((samples, 1), dtype=numpy.float32)

		for i in range(samples):

			row = map(numpy.float32, data[i][0].split(','))
			train_Y[i, 0] = row[10]
			train_X[i, :] = numpy.asarray(row[1: 9])

		test_X = train_X[int(3*samples/4)+1: , :]
		test_X = numpy.append(numpy.ones((test_X.shape[0], 1)), test_X, axis=1)
		test_Y = train_Y[int(3*samples/4)+1: , :]

		train_X = train_X[:int(3*samples/4), :]
		train_X = numpy.append(numpy.ones((train_X.shape[0], 1)), train_X, axis=1)
		train_Y = train_Y[:int(3*samples/4), :]

		return train_X, train_Y, test_X, test_Y

def train_weights(data, labels, margin, ep):
	
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
				error = numpy.matmul(
								numpy.transpose(weights),
								misclassified
								)
				Z = misclassified * (margin - error[0, 0])/(linalg.norm(misclassified)**2)
				weights = numpy.add(weights, Z)
			prev_i = i

		epoch += 1

	return weights, (1-len(prev_i)/data.shape[0])*100

def test_weights(weights, data, labels, margin):
	
	labels -= 3
	errors = numpy.matmul(data, weights)
	signs = numpy.multiply(errors, labels)
	i, j = numpy.where(signs < margin)	

	return (1-i.shape[0]/labels.shape[0])*100

if __name__ == '__main__':
	
	start_time = time.time()
	train = sys.argv[1]
	test = sys.argv[2]
	epoch = [10, 100, 1000, 10000, 100000]
	for ep in epoch:
		for margin in range(10):
	
			print "Margin: ", margin
			train_X, train_Y, test_X, test_Y = load_data(train)
			weights, train_accuracy = train_weights(train_X, train_Y, margin, ep)
			print "Train Accuaracy: ", train_accuracy
			test_accuracy = test_weights(weights, test_X, test_Y, margin)
			print "Test Accuracy: ", test_accuracy

	print "----Time Elapsed:---- ",(time.time()-start_time)
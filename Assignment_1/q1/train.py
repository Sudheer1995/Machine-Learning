#!/usr/bin/env

import numpy

def train_weights(data, labels):

	epoch = 0
	labels = 2*labels - 1
	data = numpy.multiply(labels, data)
	weights = numpy.ones((data.shape[1], 1), dtype=numpy.float32)
	while epoch < 10000:

		errors = numpy.matmul(data, weights)
		i, j = numpy.where(errors < 0)
		print i

		for k in i:
			
			misclassified = data[k, :].reshape(weights.shape[0], 1)
			weights = numpy.add(weights, misclassified)

		epoch += 1
	
	return weights

if __name__ == '__main__':
	

	X = numpy.load('train_X.npy')
	Y = numpy.load('train_Y.npy')
	weights = train_weights(X, Y)
	
	print weights.shape
	numpy.save('weights', weights)
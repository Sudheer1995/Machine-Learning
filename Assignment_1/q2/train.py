#!/usr/bin/env

from numpy import linalg
import numpy

def train_weights(data, labels):
	
	epoch = 0
	bias = 0
	labels -= 3
	weights = numpy.random.rand(data.shape[1], 1)
	data = numpy.multiply(labels, data)
	while epoch < 100000:
		
		errors = numpy.matmul(data, weights) + bias
		i, j = numpy.where(errors < 0)
		print i

		for k in i:
			
			misclassified = data[k, :].reshape(weights.shape[0], 1)
			error = numpy.matmul(
							numpy.transpose(weights),
							misclassified
							)
			Z = misclassified * 1 # (bias - error)/(linalg.norm(misclassified)**2)
			weights = numpy.add(weights, Z)

		epoch += 1

	return weights

if __name__ == '__main__':
	
	X = numpy.load('train_X.npy')
	Y = numpy.load('train_Y.npy')
	weights = train_weights(X, Y)
	print weights
	numpy.save('weights.npy', weights)

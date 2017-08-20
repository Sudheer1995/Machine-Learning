#!/usr/bin/env

from __future__ import division
import numpy

def test_weights(weights, data, labels):
		
	errors = numpy.matmul(data, weights)
	labels = 2*labels-1
	signs = numpy.multiply(errors, labels)
	i, j = numpy.where(signs < 0)	

	return (1-i.shape[0]/labels.shape[0])*100


if __name__ == '__main__':
	
	X = numpy.load('test_X.npy')
	Y = numpy.load('test_Y.npy')
	W = numpy.load('weights.npy')

	accuracy = test_weights(W, X, Y)
	print accuracy

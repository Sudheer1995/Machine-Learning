#!/usr/bin/env 

from __future__ import division
from collections import Counter
import sys
import os
import numpy as np
import numpy.matlib
import operator

"""Feel free to add any extra classes/functions etc as and when needed.
This code is provided purely as a starting point to give you a fair idea
of how to go about implementing machine learning algorithms in general as
a part of the first assignment. Understand the code well"""


class FeatureVector(object):
	def __init__(self, vocabsize, numdata):
		self.vocabsize = vocabsize
		self.X =  np.zeros((numdata,self.vocabsize), dtype=np.int)
		self.Y =  np.zeros((numdata,1), dtype=np.int)

	def make_featurevector(self, file_num, input, classid, vocabulary):
		"""Takes input the documents and outputs the feature vectors as X and classids as Y."""
		D = {key: 0 for key in vocabulary}
		for word in input:	
			try:
				D[word] += 1
			except KeyError as e:
				pass

		# total = sum(D.values())
		total = 1
		if not total == 0:
			tmp = [freq/total for freq in D.values()]
		else:
			tmp = [0]*len(D.values())
		self.X[file_num, :] = numpy.asarray(tmp).reshape(1, len(D.values()))
		self.Y[file_num, 0] = numpy.asarray(classid).reshape(1, 1)

class KNN(object):
	def __init__(self, trainVec, testVec):
		self.X_train = trainVec.X
		self.Y_train = trainVec.Y
		self.X_test = testVec.X
		self.Y_test = testVec.Y
		self.metric = Metrics('accuracy')

	def classify(self, K):
		"""Takes input X_train, Y_train, X_test and Y_test and displays the accuracies."""
		prediction = numpy.zeros((self.X_test.shape[0], 1), dtype=numpy.int)
		for i in range(self.X_test.shape[0]):
			A = self.X_test[i, :].reshape(1, self.X_test.shape[1])
			T = numpy.matlib.repmat(A, self.X_train.shape[0], 1)
			distances = numpy.sum(numpy.square(T - self.X_train), axis=1)
			indices = numpy.argsort(distances)
			prediction[i, 0] = self.Y_train[self.get_label(K, distances, indices), 0]


	def get_label(self, K, distances, indices):
		"""get closest & frequent cluster label"""
		circle = dict.fromkeys(indices[: K], 0)
		for index in indices[: K]:
			if not distances[index] == 0:
				circle[index] += 1/distances[index]

		return max(circle.iteritems(), key=operator.itemgetter(1))[0]	

class Metrics(object):
	def __init__(self, metric):
		"""initialises score metric"""
		self.metric = metric

	def score(self, y_test, y_pred):
		"""gets the score metric as per choice"""
		self.get_confmatrix(y_pred, y_test)
		if self.metric == 'accuracy':
			return self.accuracy()
		elif self.metric == 'f1':
			return self.f1_score()

	def get_confmatrix(self, y_pred, y_test):
		"""Implements a confusion matrix"""
		self.conf_matrix = numpy.zeros((y_pred.shape[0], y_pred.shape[0]), dtype=numpy.int)
		for i in range(y_pred.shape[0]): 	
			self.conf_matrix[y_pred[i, 0]-1, y_test[i, 0]-1] += 1

	def accuracy(self):
		"""Implements the accuracy function"""
		correct = 0
		total = numpy.sum(self.conf_matrix) 
		for i in range(10):
			correct += self.conf_matrix[i, i]
		self.accuracy = (correct/total)*100
		
		return self.accuracy


	def f1_score(self):
		"""Implements the f1-score function"""
		predicted = numpy.sum(self.conf_matrix, axis=0) 
		self.recall = [self.conf_matrix[i, i]/predicted[i] for i in range(10)]
		class_members = numpy.sum(self.conf_matrix, axis=1)
		self.precision = [self.conf_matrix[i, i]/predicted[i] for i in range(10)]
		self.f1_score = [2*(self.precision[i]*self.recall[i])/(self.precision[i]+self.recall[i]) for i in range(10)]
		
		return self.f1_score

if __name__ == '__main__':
	
	datadir = '../datasets/q4/'
	classes = ['galsworthy/', 'galsworthy_2/', 'mill/', 'shelley/', 'thackerey/', 'thackerey_2/', 'wordsmith_prose/', 'cia/', 'johnfranklinjameson/', 'diplomaticcorr/']
	inputdir = ['train/', 'test/']

	vocab = 0
	margin = 101
	bag_of_words =  []
	for each_class in classes:
		for file in os.listdir(datadir+inputdir[0]+each_class):

			f = open(datadir+inputdir[0]+each_class+file, 'r')
			for line in f:
				bag_of_words.extend(line.split())
	
	vocab_dict = Counter(bag_of_words)
	temp = sorted(vocab_dict.items(), key=operator.itemgetter(1), reverse=True)
	vocabulary = [item[0] for item in temp]
	vocabulary = vocabulary[margin:len(vocabulary)-margin]
	vocab = len(vocabulary)

	trainsz = 0
	for each_class in classes:
		trainsz += len(os.listdir(datadir+inputdir[0]+each_class))

	testsz = 0
	for each_class in classes:
		testsz += len(os.listdir(datadir+inputdir[1]+each_class))

	trainVec = FeatureVector(vocab, trainsz)
	testVec = FeatureVector(vocab, testsz)

	for idir in inputdir:
		classid = 1
		file_num = 0
		for c in classes:
			listing = os.listdir(datadir+idir+c)
			for filename in listing:
				f = open(datadir+idir+c+filename, 'r')
				inputs = [word for line in f for word in line.split()]
				if idir == 'train/':
					trainVec.make_featurevector(file_num, inputs, classid, vocabulary)
				else:
					testVec.make_featurevector(file_num, inputs, classid, vocabulary)
				file_num += 1
			classid += 1

	K = 3
	knn = KNN(trainVec, testVec)
	knn.classify(K)
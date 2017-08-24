#!/usr/bin/env

from __future__ import division
from collections import Counter
import numpy
import csv
import sys
import math

class Node(object):
	"""docstring for Node"""
	def __init__(self, attributes):
		self.attribute = attributes
		self.child = {}

	def set_child(children):
		self.child.append(child)

class Entropy(object):
	"""Entropy object for each subset"""
	def __init__(self):
		self.positive = 0
		self.negative = 0

	def size(self):
		return self.positive+self.negative

def str_split(S, delimiter):
	"""splits the string preserving the order"""
	L = []
	word = ''
	for i in range(len(S)):
		if S[i] == delimiter:
			L.append(word)	
			word = ''
		else:
			word += S[i]
	L.append(word)
	return L

def load_data(train_file, test_file):
	"""loads data from csv file"""
	with open(train_file) as f:
		reader = csv.reader(f, delimiter=' ', quotechar='|')
		data = list(reader)
		train_data = {key: [] for key in str_split(data[0][0], ',')}
		for i in range(1, len(data)):
			values = str_split(data[i][0], ',')
			for key, value in zip(str_split(data[0][0], ','), values):
				train_data[key].append(value)
		for key in train_data.keys():
			if not key == 'sales' and not key == 'salary':	
				train_data[key] = map(float, train_data[key])

		return train_data

def entropy(data, label, typ):
	"""gets entropy of a dataset"""
	variation = 0
	if typ == 'numerical':
		left = Entropy()
		right = Entropy()
		mean = sum(data)/len(data)
		for i in range(len(data)):
			if data[i] < mean:
				left.positive += 1-label[i]
				left.negative += label[i]
			else:
				right.positive += 1-label[i]
				right.negative += label[i]
		variation = (left.positive*math.log(left.positive/left.size()) + left.negative*math.log(left.negative/left.size()) + right.positive*math.log(right.positive/right.size()) + right.negative*math.log(right.negative/right.size()))
	else:
		freq_map = {key: Entropy() for key in Counter(data).keys()}
		for i in range(len(data)):
			freq_map[data[i]].positive += 1-label[i]
			freq_map[data[i]].negative += label[i]
		for key in freq_map.keys():
			variation += freq_map[key].positive*math.log(freq_map[key].positive/freq_map[key].size()) + freq_map[key].negative*math.log(freq_map[key].negative/freq_map[key].size())
		
	return (1/len(data))*variation

def train_tree(train_data):
	"""trains decision tree"""
	for key in train_data.keys():
		if not key == 'left':
			if type(train_data[key][0]) == str:
				print entropy(train_data[key], train_data['left'], 'string')
			else:
				print entropy(train_data[key], train_data['left'], 'numerical')

if __name__ == '__main__':
	train_file = sys.argv[1]
	test_file = sys.argv[2]
	train_data = load_data(train_file, test_file)
	train_tree(train_data)
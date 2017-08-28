#!/usr/bin/env

from __future__ import division
from collections import Counter
import numpy
import csv
import sys
import math
import operator
import time

class Node(object):
	"""Node in Decision Tree"""
	def __init__(self, attributes):
		if type(attributes) == list:
			self.child = {key: None for key in attributes}
		else:
			self.mean = attributes
			self.child = {key: None for key in ['left', 'right']}

	def set_child(self, label, child):
		self.child[label] = child

	def get_child(self, label):
		if type(label) == str:
			return self.child[label]
		else:
			if label < self.mean:
				return self.child['left']
			else:
				return self.child['right']

	def get_keys(self):
		return self.child.keys()

class Entropy(object):
	"""Entropy object for each subset"""
	def __init__(self):
		self.positive = 1
		self.negative = 1

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
	if sum(train_data['left']) == 0:
		return 0

	if sum(train_data['left']) == len(train_data['left']):
		return 1
		
	print train_data.keys()
	randomness = {key: 0 for key in train_data.keys() if not key == 'left'}
	for key in train_data.keys():
		if not key == 'left':	
			if type(train_data[key][0]) == str:
				randomness[key] += entropy(train_data[key], train_data['left'], 'string')
			else:
				randomness[key] += entropy(train_data[key], train_data['left'], 'numerical')

	label = max(randomness.iteritems(), key=operator.itemgetter(1))[0]
	print label	
	time.sleep(2)
	if type(train_data[label][0]) == str:
		elements = list(set(train_data[label]))
		node = Node(elements)
		for elem in elements:
			indices = [i for i in range(len(train_data[label])) if train_data[label][i] == elem]
			next_train_data = {key: None for key in train_data.keys() if not key == label}
			for key in next_train_data.keys():
				next_train_data[key] = [train_data[key][i] for i in indices]
			node.set_child(elem, train_tree(next_train_data))
		return node	
	else:
		elements = ['left', 'right']
		mean = sum(train_data[label])/len(train_data[label])
		node = Node(mean)
		indices = [i for i in range(len(train_data[label])) if train_data[label][i] < mean]
		for elem in elements:
			next_train_data = {key: None for key in train_data.keys()}
			if elem == 'left':
				for key in next_train_data.keys():
					next_train_data[key] = [train_data[key][i] for i in indices]
			else:
				for key in next_train_data.keys():
					next_train_data[key] = [train_data[key][i] for i in range(len(train_data[label])) if not i in indices]	
			node.set_child(elem, train_tree(next_train_data))
		return node


if __name__ == '__main__':
	train_file = sys.argv[1]
	test_file = sys.argv[2]
	train_data = load_data(train_file, test_file)
	root = train_tree(train_data)
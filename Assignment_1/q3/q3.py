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
	def __init__(self, name, attributes, typ='node'):
		"""initialises the node in a tree"""
		self.typ = typ
		self.name = name
		if self.typ == 'node':
			if type(attributes) == list:
				self.mean = None
				self.child = {key: None for key in attributes}
			else:
				self.mean = attributes
				self.child = {key: None for key in ['left', 'right']}
			self.label = None
		else:
			self.child = None
			self.label = attributes

	def set_child(self, label, child):
		"""sets a particular as child of this node"""
		self.child[label] = child

	def get_child(self, label, typ):
		"""gets child if any present else 0"""	
		if typ == 1:
			return self.child[label]
		else:
			if label < self.mean:
				return self.child['left']
			else:
				return self.child['right']

	def get_type(self):
		"""returns type of node"""
		return self.typ

	def get_label(self):
		"""returns label given to the node"""
		return self.label

	def get_name(self):
		"""returns name of the node"""
		return self.name

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
			for key, value in zip(str_split(data[0][0], ','), str_split(data[i][0], ',')):
				train_data[key].append(value)
		for key in train_data.keys():
			if not key == 'sales' and not key == 'salary':	
				train_data[key] = map(float, train_data[key])

	with open(test_file) as f:
		reader = csv.reader(f, delimiter=' ', quotechar='|')
		data = list(reader)
		test_data = {key: [] for key in str_split(data[0][0], ',')}
		for i in range(1, len(data)):
			for key, value in zip(str_split(data[0][0], ','), str_split(data[i][0], ',')):
				test_data[key].append(value)
			for key in test_data.keys():
				if not key == 'sales' and not key == 'salary':
					test_data[key] = map(float, test_data[key])
	
	return train_data, test_data

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

def train_tree(data_type, train_data, depth, max_depth):
	"""trains decision tree"""
	if len(train_data['left']) <= 1:
		leaf = Node('leaf', train_data['left'][0], 'leaf')
		return leaf

	if depth == max_depth:
		if sum(train_data['left']) > len(train_data['left'])-sum(train_data['left']):
			leaf = Node('leaf', 1, 'leaf')
		else:
			leaf = Node('leaf', 0, 'leaf')	
		return leaf

	randomness = {key: 0 for key in train_data.keys() if not key == 'left'}
	for key in train_data.keys():
		if not key == 'left':	
			if len(train_data[key]) > 0:
				if data_type[key] == 1:
					randomness[key] += entropy(train_data[key], train_data['left'], 'string')
				else:
					randomness[key] += entropy(train_data[key], train_data['left'], 'numerical')
			else:
				randomness[key] = 1000
	label = min(randomness.iteritems(), key=operator.itemgetter(1))[0]
	if data_type[label] == 1:
		elements = list(set(train_data[label]))
		node = Node(label, elements, 'node')
		for elem in elements:
			indices = [i for i in range(len(train_data[label])) if train_data[label][i] == elem]
			next_train_data = {key: None for key in train_data.keys() if not key == label}
			for key in next_train_data.keys():
				next_train_data[key] = [train_data[key][i] for i in indices]
			node.set_child(elem, train_tree(data_type, next_train_data, depth+1, max_depth))
		return node
	else:
		elements = ['left', 'right']
		mean = sum(train_data[label])/len(train_data[label])
		node = Node(label, mean, 'node')
		indices = [i for i in range(len(train_data[label])) if train_data[label][i] < mean]
		for elem in elements:
			next_train_data = {key: None for key in train_data.keys()}
			if elem == 'left':
				for key in next_train_data.keys():
					next_train_data[key] = [train_data[key][i] for i in indices]
			else:
				for key in next_train_data.keys():
					next_train_data[key] = [train_data[key][i] for i in range(len(train_data[label])) if not i in indices]	
			for key in next_train_data.keys():
				if next_train_data[key] == []:
					del next_train_data[key]
			if not next_train_data == {}:
				node.set_child(elem, train_tree(data_type, next_train_data, depth+1, max_depth))
			else:
				leaf = Node(label, 1, 'leaf')
				node.set_child(elem, leaf)
		return node

def test_tree(root, test_data, data_type):
		
	keys = test_data.keys()
	for i in range(len(test_data[keys[0]])):
		test = {key: test_data[key][i] for key in keys}
		temp = root
		while not temp.get_type() == 'leaf':
			key = temp.get_name()
			attribute = test[key]
			temp = temp.get_child(attribute, data_type[key])
		print int(temp.get_label())

if __name__ == '__main__':
	train_file = sys.argv[1]
	test_file = sys.argv[2]

	data_type = {
		'salary':1, 
		'satisfaction_level':0,
		'Work_accident':1, 
		'promotion_last_5years':1, 
		'sales':1, 
		'average_montly_hours':0, 
		'last_evaluation':0, 
		'number_project':0, 
		'time_spend_company':0, 
		'left':1
	}

	train_data, test_data = load_data(train_file, test_file)
	root = train_tree(data_type, train_data, 0, 500)
	test_tree(root, test_data, data_type)
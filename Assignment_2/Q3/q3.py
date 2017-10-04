from sklearn.metrics import accuracy_score
from sklearn import linear_model
from numpy import genfromtxt
from random import randint
from scipy.misc import imsave
import math
import numpy as np

def showarray(weights):
	weights = np.asarray(weights).reshape((28, 28))
	for i in range(weights.shape[0]):
		for j in range(weights.shape[1]):
			weights[i, j] = math.fabs(weights[i, j])*10000
	imsave('l1-0.01-weights.jpg', weights)

def get_data():
	X_train = genfromtxt('q3/notMNIST_train_data.csv', delimiter=',')
	y_train = genfromtxt('q3/notMNIST_train_labels.csv', delimiter=',')
	X_test = genfromtxt('q3/notMNIST_test_data.csv', delimiter=',')
	y_test = genfromtxt('q3/notMNIST_test_labels.csv', delimiter=',')
	return X_train, y_train, X_test, y_test

def train(X, y):
	model = linear_model.LogisticRegression(penalty='l1', dual=False, tol=0.001, C=0.01, 
		fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, 
		solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
	model.fit(X, y, sample_weight=None)
	return model

def test(model, X, y):
	y_pred = model.predict(X)
	return accuracy_score(y, y_pred,  normalize=True, sample_weight=None)

if __name__ == '__main__':
	n = randint(1, 1000)
	X_train, y_train, X_test, y_test = get_data()
	model = train(X_train, y_train)
	score = test(model, X_test, y_test)
	showarray(model.coef_)
	print "l1 0.01" 
	print "--Accuracy--: ", score
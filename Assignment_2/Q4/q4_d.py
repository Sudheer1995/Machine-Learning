from sklearn import linear_model
from sklearn.metrics import accuracy_score
import csv
import sys
import numpy as np

def get_data(train_set, test_set):
	with open(train_set) as train_file:
		train = []
		train_data = csv.reader(train_file, delimiter=',', quotechar='|')	
		for row in train_data:
			train.append([float(elem) for elem in row])
		train = np.asarray(train)
		train_X = train[:, :-1]
		train_Y = train[:, -1]

	with open(test_set) as test_file:
		test = []	
		test_data = csv.reader(test_file, delimiter=',', quotechar='|')
		for row in test_data:
			test.append([float(elem) for elem in row])
		test = np.asarray(test)
		test_X = test[:, :-1]

	return train_X, train_Y, test_X

def train(X, y):
	# Simple
	lin_reg = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
	lin_reg.fit(X, y)
	return lin_reg

def test(model, X):
	y_pred = model.predict(X)
	y_pred = np.asarray([1 if y_pred[i] > 0.5 else 0 for i in range(y_pred.shape[0])])
	for i in range(y_pred.shape[0]):
		print y_pred[i]

if __name__ == '__main__':

	train_file = sys.argv[1]
	test_file = sys.argv[2]
	train_X, train_Y, test_X = get_data(train_file, test_file)
	model = train(train_X, train_Y)
	test(model, test_X)
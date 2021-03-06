from sklearn.metrics import accuracy_score
from sklearn import svm
from keras.models import Model, load_model
from keras.utils import np_utils
import sys
import cPickle
import numpy as np

def unpickle(file, train=False):
    with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
		if train == True:
			if 'label_names' in dict.keys():
				return dict['label_names']
			else:
				return dict['data'], dict['labels']
		else:
			return dict['data']

def get_data(X):
	model = load_model('NN1.h5')
	features = Model(model.input, model.layers[8].output)
	output = features.predict(X)
	return output

def train(X, y):
	clf = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, 
		shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, 
		verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
	clf.fit(get_data(X), y, sample_weight=None)
	return clf

def test(clf, X, label_names):
	y_pred = clf.predict(get_data(X))
	for i in range(y_pred.shape[0]):
		print label_names[y_pred[i]]

if __name__ == '__main__':

	data_dir = sys.argv[1]
	test_file = sys.argv[2]
	label_names = unpickle("batches.meta", True)
	for i in range(5):
		data, labels = unpickle(str(data_dir)+"data_batch_"+str(i+1), True)
		X = np.asarray(data[:, :]).reshape(10000, 32, 32, 3)
		Y = labels[:]
		clf = train(X, Y)

	data = unpickle(test_file)
	X = np.asarray(data[:, :]).reshape(10000, 32, 32, 3)
	test(clf, X, label_names)
from sklearn.metrics import accuracy_score
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as  K
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

def train(x_train, y_train, input_shp, activation_function='relu'):
	
	sample = x_train.shape[0]
	features = 1024
	classes = y_train.shape[1]
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(7, 7), activation='relu', input_shape=input_shp))
	model.add(Conv2D(64, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', 
		gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, 
		gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(features, activation=activation_function))
	model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', 
		gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, 
		gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
	model.add(Dense(features, activation=activation_function))
	model.add(Dropout(0.5))
	model.add(Dense(classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
	model.summary()
	model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.1, shuffle=True)
	model.save('NN1.h5')

def test(x_test, label_names):
	model = load_model('NN1.h5')
	labels = np.argmax(model.predict(x_test), axis=1)
	predictions = [label_names[label] for label in labels]
	for prediction in predictions:
		print prediction

if __name__ == '__main__':

	data_dir = sys.argv[1]
	test_file = sys.argv[2]
	label_names = unpickle("batches.meta", train=True)
	for i in range(5):
		data, labels = unpickle(str(data_dir)+"data_batch_"+str(i+1), train=True)
		if i == 0:
			X_train = np.asarray(data[:, :]).reshape(10000, 32, 32, 3)
			Y_train = np_utils.to_categorical(labels[:], len(label_names))
		else:
			X_train = np.append(X_train, np.asarray(data[:, :]).reshape(10000, 32, 32, 3), axis=0)
			Y_train = np.append(Y_train, np_utils.to_categorical(labels[:], len(label_names)), axis=0)
	train(X_train, Y_train, (32, 32, 3), activation_function='relu')

	data = unpickle(test_file)
	X_test = np.asarray(data[:, :]).reshape(10000, 32, 32, 3)
	test(X_test, label_names)
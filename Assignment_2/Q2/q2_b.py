from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as  K
import cPickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    if 'label_names' in dict.keys():
    	return dict['label_names']
    else:
    	return dict['data'], dict['labels']

def train(x_train, y_train, input_shp, activation_function='relu'):
	
	sample = x_train.shape[0]
	features = x_train.shape[1]
	classes = y_train.shape[1]
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shp))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
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
	model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
	model.summary()
	model.fit(x_train, y_train, epochs=10, batch_size=128)
	model.save('NN.h5')

def test(x_test, y_test):
	model = load_model('NN.h5')
	loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
	print loss_and_metrics

if __name__ == '__main__':

	label_names = unpickle("batches.meta")
	
	data, labels = unpickle("data_batch_1")
	X_train = np.asarray(data[:, :])
	X_train = X_train.reshape(10000, 32, 32, 3)
	Y = labels[:]
	Y_train = np_utils.to_categorical(Y, len(label_names))
	train(X_train, Y_train, (32, 32, 3), activation_function='relu')
	
	data, labels = unpickle("data_batch_5")
	X_test = np.asarray(data[:, :])
	X_test = X_test.reshape(10000, 32, 32, 3)
	Y = labels[:]	
	Y_test = np_utils.to_categorical(Y, len(label_names))
	test(X_test, Y_test)

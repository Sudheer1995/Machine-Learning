import csv
import numpy

def load_data(train_file, test_file):

	with open(train_file) as f:

		reader = csv.reader(f, delimiter=' ', quotechar='|')
		data = list(reader)
		samples = len(data)

		train_X = numpy.zeros((samples, 784), dtype=int)
		train_Y = numpy.zeros((samples, 1), dtype=int)

		for i in range(samples):

			row = map(int, data[i][0].split(','))
			train_Y[i, 0] = row[0]
			train_X[i, :] = numpy.asarray(row[1: ])

		print train_X.shape
		print train_Y.shape

		numpy.save('train_X.npy', train_X)
		numpy.save('train_Y.npy', train_Y)

	with open(test_file) as f:

		reader = csv.reader(f, delimiter=' ', quotechar='|')
		data = list(reader)
		samples = len(data)

		test_X = numpy.zeros((samples, 784), dtype=int)
		test_Y = numpy.zeros((samples, 1), dtype=int)

		for i in range(samples):

			row = map(int, data[i][0].split(','))
			test_Y[i, 0] = row[0]
			test_X[i, :] = numpy.asarray(row[1: ])

		print test_X.shape
		print test_Y.shape

		numpy.save('test_X.npy', test_X)
		numpy.save('test_Y.npy', test_Y)

if __name__ == '__main__':
	
	load_data('../datasets/dataset_q1/mnist_train.csv', 
		'../datasets/dataset_q1/mnist_test.csv')
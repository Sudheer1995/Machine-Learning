import csv
import numpy

def load_data(train_file):

	with open(train_file) as f:

		reader = csv.reader(f, delimiter=' ', quotechar='|')
		data = list(reader)
		samples = len(data)

		train_X = numpy.zeros((samples, 9), dtype=int)
		train_Y = numpy.zeros((samples, 1), dtype=int)

		for i in range(samples):

			row = map(int, data[i][0].split(','))
			train_Y[i, 0] = row[9]
			train_X[i, :] = numpy.asarray(row[: 9])

		print train_X.shape
		print train_Y.shape

		numpy.save('train_X.npy', train_X)
		numpy.save('train_Y.npy', train_Y)

if __name__ == '__main__':
	
	load_data('../datasets/dataset_q2/breast_cancer_train.csv')
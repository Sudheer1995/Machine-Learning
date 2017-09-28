"""
Course: Statistical Methods in Artificial Intelligence (CSE471)
Semester: Fall '17
Professor: Gandhi, Vineet

Assignment 2: Check if the 2D points are linearly separable and if they
are not, implement a kernel that maps them to 3-dimensional space
where they are linearly separable. We prove they are indeed linearly
separable using a perceptron linear classifier.
Skeleton code for visualizing 2D points and writing a kernel for mapping
them to a higher dimensional space. Use the perceptron given by sklearn and
prove that the kernel mapping makes the points linearly separable, using a
simple perceptron.
This is your final submission file.

Dataset is generated manually.

Remember
--------
"""

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import argparse, os, sys

def get_input_data(filename):
    """
    Function to read the input data from the letter recognition data file.

    Parameters
    ----------
    filename: The path to input data file

    Returns
    -------
    X: The input for the SVM classifier of the shape [n_samples, n_features].
       n_samples is the number of data points (or samples) that are to be loaded.
       n_features is the length of feature vector for each data point (or sample).
    Y: The labels for each of the input data point (or sample). Shape is [n_samples,].

    """

    X = []; Y = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split()
            Y.append(line[0])
            X.append([float(x) for x in line[1:]])
    X = np.asarray(X); Y = np.asarray(Y)

    return X, Y

def calculate_accuracy(predictions, labels):
    """
    Function to calculate the accuracy for a given set of predictions and
    corresponding labels.

    Parameters
    ----------
    predictions: The predictions obtained as output from the SVM classifier
    labels: The true label values corresponding to the entries in predictions

    Returns
    -------
    accuracy: Fraction of total samples that have correct predictions (same as
    true label)

    """
    return accuracy_score(labels, predictions)

def kernelize(X, Y):
    """
    Function to map the input 2D non-linearly separable points to a higher dimension
    where they can be linearly separated using a perceptron.

    Parameters
    ----------
    X: Numpy array with the input 2D points [n_samples, 2]

    Returns
    -------
    kernelX: Numpy array with the 3D kernelized inputs [n_samples, 3]

    """

    """
    ===========================================================================
    """

    # YOUR CODE HERE

    """
    ===========================================================================
    """

    """
    Code to visualize the generated 3D data after applying the kernel function, so
    that it is easier to decide which kernel works.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color = lambda x: 'r' if x else 'b'
    mark = lambda x: 'o' if x else '^'
    for i, x, y, z in enumerate(kernelX):
        ax.scatter(x, y, z, c=color(Y[i]), marker=mark(Y[i]))
    plt.show()

    return kernelX

def perceptron(X, Y):
    """
    Function to fit a perceptron to the kernelized input data.

    Parameters
    ----------
    X: Numpy array with the input 2D data points [n_samples, 2]
    Y: Numpy array with the binary classes (either 0 or 1) [n_samples,]

    Returns
    -------
    is_linearly_separable: Boolean value indicating whether the data is
                           linearly separable or not

    """

    is_linearly_separable = False
    kernelX = kernelize(X, Y)           # Y is passed to plot the kernelized data with labels

    """
    Create a Perceptron instance and fit it to the kernelized data.
    For details on how to do this in scikit-learn, refer:
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
    ==========================================================================
    """

    # YOUR CODE GOES HERE

    """
    ==========================================================================
    """

    """
    Calculates training accuracy. Replace predictions and labels with your
    respective variable names.
    """
    train_accuracy = calculate_accuracy(train_predictions, train_labels)
    print "Training Accuracy: %.4f" % (train_accuracy)

    if train_accuracy == 1:
        is_linearly_separable = True

    """
    Code to plot the 3D decision surface that separates the points. Visualization
    will help in understanding how the kernel has performed.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color = lambda x: 'r' if x else 'b'
    mark = lambda x: 'o' if x else '^'
    for i in range(kernelX.shape[0]):
        x, y, z = kernelX[i]
        ax.scatter(x, y, z, c=color(Y[i]), marker=mark(Y[i]))

    W = clf.coef_[0]                # 3D normal
    intercept = clf.intercept_[0]   # Distance from origin

    xx, yy = np.meshgrid(kernelX, Y)
    """
    Derived from a*x + b*y + c*z + d = 0. (a, b, c) is the normal and we know
    x and y values. d is the distance from origin, or the intercept. Hence, value
    of z => (-a*x - b*y - d) / c.
    Source: https://stackoverflow.com/questions/3461869/plot-a-plane-based-on-a-normal-vector-and-a-point-in-matlab-or-matplotlib
    """
    zz = ((-W[0] * xx) + (-W[1] * yy) + (-intercept)) * 1. / W[2]

    ax.plot_surface(xx, yy, zz, rstride=10, cstride=10, color='g', antialiased=False,
            linewidth=0, shade=False)
    plt.show()

    return is_linearly_separable


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None,
            help='path to the directory containing the dataset file')

    args = parser.parse_args()
    if args.data_dir is None:
        print "Usage: python kernel_trick_perceptron.py --data_dir='<dataset dir path>'"
        sys.exit()
    else:
        filename = os.path.join(args.data_dir, 'linearly_inseparable.data')
        try:
            if os.path.exists(filename):
                print "Using %s as the dataset file" % filename
        except:
            print "%s not present in %s. Please enter the correct dataset directory" % (filename, args.data_dir)
            sys.exit()

    """
    Get the input data using the provided function. Store the X and Y returned
    as X_data and Y_data. Use filename found above as the input to the function.
    ==========================================================================
    """

    # YOUR CODE GOES HERE

    """
    ==========================================================================
    """

    """
    We see if the perceptron can learn a decision boundary to give 100% accuracy.
    On succeeding, we say the kernel tricks works; else, we need to change the
    kernel function.
    ==========================================================================
    """
    if(perceptron(X, Y)):
        print "Data is linearly separable using the current kernel."
    else:
        print "Data is still not linearly separable using the current kernel."

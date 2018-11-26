# -*- coding: utf-8 -*-
"""Perceptron implentation for the ML course 2018/2019

This module implements the basic algorithm for a perceptron classifier
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Perceptron():

    def __init__(self, learning_rate, num_epochs):
        """
        :param learning_rate: is the magnitude at which we increase or decrease our weights during each iteration of training
        :param num_epochs:  the number of times we’ve iterated through the entire training set.
              we were able to establish weights to classify all of our inputs, but we continued iterating,
              to be sure that our weights were tried on all of our inputs.
        """
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = []

    def __predict(self, data):
        activation_function = lambda inputs, weights: 1 if np.dot(weights, inputs) > 0 else 0
        return activation_function(data, self.weights)

    def predict(self, data):
        results = [self.__predict(i) for i in data]
        return results

    def update_weights(self, enlarged_train_data, all_differences):
        for input, difference in zip(enlarged_train_data, all_differences):
            self.weights+=self.learning_rate*difference*input


    def print_desicion_function(self,coefs,intersect,xA,yA,xB,yB,x):
        fsize = 14

        # The decision function is computed using the coefficients and intersect learned
        # by the algorithm
        decision_function = (-intersect - coefs[0] * x) / coefs[1]

        fig = plt.figure()

        # The decision function is plotted
        plt.plot(x, decision_function, 'y*', lw=4)

        # The points from the two classes are plotted
        plt.plot(xA, yA, 'ro', lw=4)
        plt.plot(xB, yB, 'bs', lw=4)

        blue_patch = mpatches.Patch(color='blue', label='Class I')
        red_patch = mpatches.Patch(color='red', label='Class II')
        plt.legend(handles=[blue_patch, red_patch])
        plt.xlabel(r'$x$', fontsize=fsize)
        plt.ylabel(r'$y$', fontsize=fsize)

        plt.show()

    def train(self, training_data, labels):

        # Number of instances in the dataset
        N = training_data.shape[0]
        # we will deal whit the bias adding an always active input
        enlarged_train_data = np.hstack((training_data, np.ones((N, 1))))
        # Number of variables plus the bias
        n = enlarged_train_data.shape[1]

        print("Number of instances: " + str(N) + ". Number of variables: " + str(
            n - 1) + ". Plus one variable that represents the bias.")
        # Weights are initialized
        self.weights = np.random.rand(n)
        error = 0
        epoch = 0
        while epoch == 0 or (error > 0 and epoch < self.num_epochs):
            # The perceptron is used to make predictions
            predicted = self.predict(enlarged_train_data)
            # For each instance, we compute the difference between the prediction and the class
            all_differences = labels - predicted
            # Using the differences the weights are updated
            self.update_weights(enlarged_train_data, all_differences)
            epoch = epoch + 1
            # We compute the error
            error = sum(all_differences ** 2) / N
            print("Epoch :" + str(epoch) + " Error: " + str(error) + " Weights: ", self.weights)
            # fname = "perceptron_"+str(epoch)+".png"
            # fig.savefig(fname)

        return error, predicted, self.weights


if __name__ == '__main__':
    # Points in Class A
    xA = 20 * np.random.rand(50)
    shiftA = 20 * np.random.rand(50)
    yA = (4 + xA) / 2.0 - shiftA - 0.1

    # Points in Class B
    xB = 20 * np.random.rand(50)
    shiftB = 20 * np.random.rand(50)
    yB = (4 + xB) / 2.0 + shiftB + 0.1

    # We define our set of observations (the union of points from the two classes)
    # We concatenate the vectors
    x = np.hstack((xA, xB)).reshape(-1, 1)
    #print("x vector\n", x)
    y = np.hstack((yA, yB)).reshape(-1, 1)
    #print("y vector\n", y)
    x_data = np.hstack((x, y))
    #print("data vector\n", x_data)

    # In the vector of target values, the first 50 instances belong to one class and the next 50 instances belong
    # to the other class
    target_class = np.hstack((np.ones((len(xA))), np.zeros((len(xB)))))
    perceptron = Perceptron(learning_rate=0.1,num_epochs=10)
    perceptron.train(x_data,target_class)
    perceptron.print_desicion_function(perceptron.weights[0:2], perceptron.weights[ 2], xA, yA, xB, yB, x)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 16:41:45 2017
Loads MNIST data.
Creates MLP object instance.
Trains MLP model on MNIST training set.
@author: bettmensch
"""

from mnist_data import get_mnist_data
from TwoLP_classes import TwoLP
import argparse
import os

def main():
    """Takes the MLP class for a test drive"""
    
    parser = argparse.ArgumentParser(description = 'Train MLP on MNIST dataset')
    
    parser.add_argument('-mi', '--max_iter',
                        required = False,
                        type = int,
                        help = 'Number of iterations for stochastic gradient descent',
                        800)
    parser.add_argument('-il', '--input_layer_size',
                        type = int,
                        required = False,
                        help = 'Set the size of the input layer.',
                        default = 784)
    parser.add_argument('-hl_1', '--hl_1_layer_size',
                        type = int,
                        required = False,
                        help = 'Set the size of the first hidden layer.',
                        default = 550)
    parser.add_argument('-hl_2', '--hl_2_layer_size',
                        type = int,
                        required = False,
                        help = 'Set the size of the second hidden layer.',
                        default = 150)
    parser.add_argument('-ol', '--output_layer_size',
                        type = int,
                        required = False,
                        help = 'Set the size of the output layer.',
                        default = 10)
    parser.add_argument('-bs', '--batch_size',
                        required = False,
                        type = int,
                        default = 100,
                        help = 'Set the size of the random samples chosen in each stochastic gradient computation.')
    parser.add_argument('-lr', '--learning_rate',
                        required = False,
                        type = float,
                        help = 'Set the learning rate for the stochastic gradient descent.',
                        default = 0.03)
    parser.add_argument('-rp', '--reg_param',
                        required = False,
                        type = float,
                        help = 'Set the regularization parameter for training to manage overfitting.',
                        default = 0.0)
    parser.add_argument('-nf', '--norm_factor',
                        required = False,
                        type = float,
                        help = 'Set the normalization factor. Reasonable range is (0,3).',
                        default = 1.0)
    
    opts = vars(parser.parse_args())
    
    max_iter = opts['max_iter']
    il = int(opts['input_layer_size'])
    hl_1 = int(opts['hl_1_layer_size'])
    hl_2 = int(opts['hl_2_layer_size'])
    ol = int(opts['output_layer_size'])
    batch_size = opts['batch_size']
    learning_rate = opts['learning_rate']
    reg_param = opts['reg_param']
    norm_factor = opts['norm_factor']
    
    print("Getting data...")
    X_train, y_train, X_test, y_test = get_mnist_data(norm_factor)
    print("Got data. Creating and training model...")

    model = TwoLP(il, hl_1, hl_2, ol)
    
    #input("Press enter to visualize first ten samples.")
    #twolp.visualize_input(X_train)
    
    input("Press enter to train model.")

    model.train(X_train = X_train,
              y_train = y_train,
              batch_size = batch_size,
              max_iter = max_iter,
              learning_rate = learning_rate,
              reg_param = reg_param)
    
    input("Model trained. Press enter to evaluate model on training data.")
    print("Evaluating...")
    
    train_acc = model.test(X_test = X_train,
                           y_test = y_train)
    
    print("Training accuracy is %s." %train_acc)
    
    input("Press enter to evaluate model on testing data.")
    print("Evaluating...")
    
    model.test(X_test = X_test,
               y_test = y_test)
    
    save = input("Do you want to save the model? [y/n]")
     
    if save.lower() == 'y':
        directory = os.getcwd()
        model_name = input("Please enter a save name for the trained model.")
        model.dump(directory, model_name)
        
    input("Press enter to quit.")
    
if __name__ == "__main__":
    main()
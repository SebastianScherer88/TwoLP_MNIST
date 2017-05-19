#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:27:55 2017

@author: bettmensch
"""
import os
import dill
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer



def visualize_matrix(matrix):
    """Takes a square matrix and visualizes it by plotting its color coded entries."""

    # make a color map of fixed colors
    cmap = matplotlib.colors.ListedColormap(['white','black'])
    bounds=[-6,0.2,6]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # tell imshow about color map so that only set colors are used
    img = matplotlib.pyplot.imshow(matrix,interpolation='nearest',
                                   cmap = cmap,norm=norm)

    # make a color bar
    matplotlib.pyplot.colorbar(img,cmap=cmap,
                               norm=norm,boundaries=bounds,ticks=[-5,0,5])

    matplotlib.pyplot.show()
    

def matrix_similarity(matrix_a, matrix_b):
    """Takes two matrices and makes pairwise row comparisons. Calculates
    the precentage of identical rows for over all pairs.
    Returns that percentage."""
    
    m, n = matrix_a.shape
    
    same = 0
    
    for row_a_i, row_b_i in zip(matrix_a, matrix_b):
        list_a_i = [row_a_i.item(i) for i in range(n)]
        list_b_i = [row_b_i.item(i) for i in range(n)]
        same += (list_a_i == list_b_i)
        
    return same / m

def L_2_norm(matrix):
    """Takes a matrix and calculates its L^2 norm.
    Returns the matrix's L^2 norm."""
    return np.linalg.norm(matrix)

def rand_mat(m_rows,n_columns):
    """Takes to integers and returns random matrix with said dimensions."""
    epsilon = math.sqrt(6) / math.sqrt(m_rows + n_columns)
    high = epsilon * np.matrix(np.ones((m_rows, n_columns)))
    low = - high
    
    return np.matrix(np.random.uniform(low, high))

def tanh_mat(matrix):
    """Takes a matrix and applies the tanh function element_wise.
    Returns transformed matrix."""
    return np.tanh(matrix)
    
def d_tanh_mat(matrix):
    """Takes a matrix and applies the tanh function's derivative element_wise.
    Returns transformed matrix."""
    return 1 - np.multiply(tanh_mat(matrix), tanh_mat(matrix))
    
def softmax(vector):
    """Takes a vector and applies the softmax.
    Returns transformed vector."""
    return np.exp(vector) / np.sum(np.exp(vector))

def softmax_mat(matrix):
    """Takes a matrix and applies the softmax to each row.
    Returns transformed matrix."""
    return np.matrix(np.vstack([softmax(row) for row in matrix]))
    
def softmax_entropy(t,y):
    """Takes a target vector t and a prediction vector y and calculates the 
    cross entropy.
    Returns the cross entropy (scalar)."""
    return np.sum(np.multiply(t,np.log(y)))

def softmax_entropy_mat(T,Y):
    """Takes a matrix T where "one row = one target label".
    Takes a matrix Y where "one row = one predicted label".
    Returns the total cross entropy for a softmax output layer."""
    m = T.shape[0]
    return - sum([softmax_entropy(t,y) for t,y in zip(T,Y)]) / m
            
class TwoLP(object):
    """MultiLayerPerceptron class"""
    def __init__(self, IL, HL_1, HL_2, OL):
        """Takes an iterable layers containing the layer_sizes. The length of
        the iterable determines the depth of the MLP."""
        
        # add layer sizes for easier reference later
        self.IL = IL
        self.HL_1 = HL_1
        self.HL_2 = HL_2
        self.OL = OL
        
        # add layer value storages
        self.B_0 = None
        self.A_1 = None
        self.B_1 = None
        self.A_2 = None
        self.B_2 = None
        self.A_3 = None
        self.B_3 = None
        # create parameter attributes
        self.initialize_params()
        # create training attributes
        self.training_data = None
        self.training_params = None
        self.trained = False
        # create preprocessing encoding and decoding objects
        self.lb = LabelBinarizer()
        
        print("Model object created.")
                
    def train(self,X_train, y_train,
              learning_rate = 0.05,
              tolerance_threshold = 1.0e-06,
              max_iter = 300,
              batch_size = 100,
              reg_param = 0):
        """Takes a matrix of training samples of the form "one row = one sample"
        and a matrix of training labels of the form "one row = one label".
        Optional parameters:
            learning rate for gradient descent (scalar)
            tolerance threshold for break criteria (scalar)
            maximal number of iterations(scalar)
        Trains model weights and attaches optimal weights to model."""
        
        # temporarily store training & test attributes
        Y_train = self.lb.fit_transform(y_train)
        
        self.training_data = X_train, Y_train
        
        self.training_params = batch_size, reg_param
        
        #print("Training mode is %s . Optimizing model parameters..." %self.training_mode[0])
        init_params = self.mats_to_vec(self.W_1, self.W_2, self.W_3, self.f_1, self.f_2, self.f_3)
                                       
        opt_w, error_history = self.optimize_params(self.cost,init_params,
                                     lamda = learning_rate,
                                     tol = tolerance_threshold,
                                     max_iter = max_iter)

        (self.W_1, self.W_2, self.W_3, self.f_1, self.f_2, self.f_3) = self.vec_to_mats(opt_w)
        
        self.trained = True
        
        print("Model parameters optimized. Training run complete.")
        
        self.visualize_training(error_history)
        
    def initialize_params(self):
        """Randomly initializes the weight matrices. Refers to layer_sizes 
        attribute and returns iterable of randomly initialized weights matrices.
        Randomly initializes the bias vectors. Refers to layer_sizes attribute
        and returns iterable of randomly initialized bias vectors."""
        
        self.W_1 = rand_mat(self.IL,self.HL_1)
        self.W_2 = rand_mat(self.HL_1,self.HL_2)
        self.W_3 = rand_mat(self.HL_2,self.OL)
        self.f_1 = np.matrix(np.ones((1,self.HL_1))) / 2
        self.f_2 = np.matrix(np.ones((1,self.HL_2))) / 2
        self.f_3 = np.matrix(np.ones((1,self.OL))) / 2
    
    def mats_to_vec(self, W_1, W_2, W_3, f_1, f_2, f_3):
        """Takes a pair of iterables containing the weights matrices and biases
        row vectors, respectively."""
        V_1 = W_1.reshape(1,-1)
        V_2 = W_2.reshape(1,-1)
        V_3 = W_3.reshape(1,-1)
        
        params = np.matrix(np.hstack([V_1, V_2, V_3, f_1, f_2, f_3]))
        
        return params
    
    def vec_to_mats(self,params):
        """Takes a long vector of mixed weights and biases parameters and returns
        two iterables of weights and biases, respectively."""
        
        f_3 = params[0,-self.OL:]
        f_2 = params[0,-self.OL - self.HL_2:-self.OL]
        f_1 = params[0,-self.OL - self.HL_2 - self.HL_1:-self.OL - self.HL_2]
        
        W_1 = params[0,:self.IL * self.HL_1].reshape(self.IL,self.HL_1)
        W_2 = params[0,self.IL * self.HL_1:
                        self.IL * self.HL_1 + self.HL_1 * self.HL_2].reshape(self.HL_1,self.HL_2)
        W_3 = params[0,self.IL * self.HL_1 + self.HL_1 * self.HL_2:
                        self.IL * self.HL_1 + self.HL_1 * self.HL_2 + self.HL_2 * self.OL].reshape(self.HL_2, self.OL)
        
            
        return (W_1, W_2, W_3, f_1, f_2, f_3)
            
    def optimize_params(self,f,x_0,lamda,tol,max_iter):
        """Takes a function to be optimized. The function must return a tuple
        of the form (f,Df), where f is the scalar function value and Df is the
        function's gradient.
        Takes the learning rate 'lamda' (scalar), a tolerance threshold 'tol'
        for the cauchy break criteria.
        Takes a maximum number of iterations 'max_iter' (scalar).
        Returns x_opt, the value at which f attains its minimum."""
        
        iteration = 0
        x = x_0
        f_x, Df_x = f(x)
        temp = f_x + 1
        f_history = []
        
        while iteration <= max_iter and abs(temp - f_x) >= tol:
            temp = f_x
            x = x - lamda * Df_x
            f_x, Df_x = f(x)
            
            iteration += 1
            f_history.append(f_x)
            
            print("Iteration ", iteration)
            print("Loss: ", f_x)
        
        return x, f_history #, train_acc_history, test_acc_history
        
    def cost(self,params):
        """Takes a row vector of model parameters (weights and biases).
        Returns the cost function's value L(params) and the cost function's
        gradient DL(params)."""
                   
        batch_size, reg_param = self.training_params
        
        index_set = np.random.choice(len(self.training_data[0]),
                                     size = batch_size,
                                     replace = False)
        
        X_train = self.training_data[0][index_set]
        Y_train = self.training_data[1][index_set]
        
        (W_1, W_2, W_3, f_1, f_2, f_3) = self.vec_to_mats(params)
        
        L = self.forward_prop(X = X_train, Y = Y_train,
                              W_1 = W_1, W_2 = W_2, W_3 = W_3,
                              f_1 = f_1, f_2 = f_2, f_3 = f_3,
                              mode = 'training',
                              reg_param = reg_param)
        DL = self.back_prop(W_1 = W_1, W_2 = W_2, W_3 = W_3, Y = Y_train)
        
        return L, DL
    
    def forward_prop(self, X, W_1, W_2, W_3, f_1, f_2, f_3,
                     mode, Y = None,
                     reg_param = 0):
        """Takes a matrix of training samples where "one row = one sample".
        Takes a matrix of training lables where "one row = one sample".
        Takes an iterable of weight matrices Ws and an iterable of biases fs.
        Takes a mode, either "training" or "prediction"."""
        
        #format biases for computation purposes
        f_1s = np.vstack([f_1 for i in range(X.shape[0])])
        f_2s = np.vstack([f_2 for i in range(X.shape[0])])
        f_3s = np.vstack([f_3 for i in range(X.shape[0])])
      
        # initialize input values
        m = X.shape[0]
        self.B_0 = X
        
        # run forward prop
        self.A_1 = np.dot(self.B_0, W_1) + f_1s
        self.B_1 = tanh_mat(self.A_1)
        
        self.A_2 = np.dot(self.B_1, W_2) + f_2s
        self.B_2 = tanh_mat(self.A_2)
        
        self.A_3 = np.dot(self.B_2, W_3) + f_3s
        self.B_3 = softmax_mat(self.A_3)
        
        # return loss function value if in training mode
        if mode == 'training':
            L = softmax_entropy_mat(Y, self.B_3) + self.reg_term(W_1, W_2, W_3, m, reg_param)

            return L
        
        # return predictions only if in prediction mode
        elif mode == 'prediction':
            P = self.B_3
            
            return P
        
    def reg_term(self, W_1, W_2, W_3, m, reg_param):
        """Takes an iterable of weight matrices and calculates the L^2 norm of 
        the weights.
        Returns the L^2 norm of the weight matrices."""
        if reg_param:
            return reg_param * sum([L_2_norm(W) for W in (W_1, W_2, W_3)]) / (2 * m)
        else:
            return 0
        
    def back_prop(self, W_1, W_2, W_3,
                  Y,
                  reg_param = 0):
        """Takes a matrix of training labels where "one row = one sample".
        Takes an iterable of weight matrices and an iterable of biases fs."""
        # get batch size
        m = Y.shape[0]
        
        # start backprop
        Delta_3 = 1 / m * (self.B_3 - Y)
        DW_3 = np.dot(self.B_2.T, Delta_3) + reg_param * W_3 / m
        Df_3 = np.sum(Delta_3,0)
        
        Delta_2 = np.multiply(np.dot(Delta_3,W_3.T),
                              d_tanh_mat(self.A_2))
        DW_2 = np.dot(self.B_1.T, Delta_2) + reg_param * W_2 / m
        Df_2 = np.sum(Delta_2,0)
        
        Delta_1 = np.multiply(np.dot(Delta_2,W_2.T),
                              d_tanh_mat(self.A_1))
        DW_1 = np.dot(self.B_0.T, Delta_1) + reg_param * W_1 / m
        Df_1 = np.sum(Delta_1,0)
            
        # rearrange derivative matrices into parameter vector and return
        DL = self.mats_to_vec(DW_1, DW_2, DW_3, Df_1, Df_2, Df_3)
        
        return DL
    
    def visualize_training(self,L_hist):
        """Takes an iterable of Loss function errors computed between consecutive
        time steps during training.
        Plots the error history against the time steps."""
        
        plt.figure(1)
        plt.xlabel('Time steps')
        plt.ylabel('Loss function')
        plt.plot(L_hist)
        
        plt.show()
            
    def visualize_input(self, X, k = 10):
        """Takes a matrix where "one row = one sample".
        Visualizes the first 10 samples"""
        for x_i in X[:k]:
            matrix = np.reshape(x_i, (28, 28))
            visualize_matrix(matrix)
            
    def visualize_features(self):
        """Takes one weight matrix. 
        Visualizes it by color coding each of its columns."""
        m, n = self.W_3.shape
        o = int(math.sqrt(m))

        features = [np.matrix(np.reshape(column, (o, -1))) for column in self.W_3.T]
        
        for feature in features[:10]:
            visualize_matrix(feature)
            
    def predict(self, X_pred):
        """Takes a matrix of samples where "one row = one sample".
        Calculates model predictions using trained weights and forward_prop.
        Returns matrix of prediction labels where "one row = one predicted label
        vector."""
        
        # make predictions
        P = self.forward_prop(X = X_pred,
                     W_1 = self.W_1, W_2 = self.W_2, W_3 = self.W_3,
                     f_1 = self.f_1, f_2 = self.f_2, f_3 = self.f_3,
                     mode = 'prediction')
        
        y_pred = np.array([self.lb.inverse_transform(p) for p in P]).reshape(-1)
        
        return y_pred
    
    def test(self, X_test, y_test):
        """Takes a matrix of test samples where "one row = one sample".
        Takes a matrix of test labels where "one row = one label".
        Calculates predictions and evaluates accuracy based on predictions
        and test labels.
        Returns an accuracy score."""
        
        # calculate model predictions
        y_pred = self.predict(X_pred = X_test)
        
        print(confusion_matrix(y_test, y_pred, labels = self.lb.classes_))
        
        print(classification_report(y_test, y_pred, labels = self.lb.classes_))
    
    def dump(self, directory, file_name):
        """Takes a filepath (string type) and dumps the model using dill."""
        work_dir = os.getcwd()
        
        os.chdir(directory)
                            
        with open(file_name, 'wb') as dump_file:
            dill.dump(self, dump_file)
        
        os.chdir(work_dir)
        
    def clone(self):
        """Returns a copy of self."""
        
        clone = TwoLP(self.IL, self.HL_1, self.HL_2, self.OL)
        
        if not self.trained:
            return clone
        else:
            clone.W_1 = self.W_1[:]
            clone.W_2 = self.W_2[:]
            clone.W_3 = self.W_3[:]
        
            clone.f_1 = self.f_1[:]
            clone.f_2 = self.f_2[:]
            clone.f_3 = self.f_3[:]
            
            clone.lb = self.lb
            
            clone.trained = True
        
        return clone
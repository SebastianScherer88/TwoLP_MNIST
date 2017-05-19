Everything that is on here is for public use, and I encourage everyone who is interested to download, clone, edit, use and
experiment with these models at their hearts' desire! Also, feel free to let me know if there a re any bugs or improvements you
think would be great to have, and I might just get around and update those, too.
As far as dependencies go, I have tried to make it as self-contained as possible. You will need, depending on the model, some of the following packages/moduels installled:

- numpy
- sklearn
- and whatever else it says at the top of the scripts.

A bit about the model on here:

The TwoLP_MNIST folder contains a from-scratch implementation of a two layer (i.e. two hidden layers, plus one input and one output layer) feed forward neural net, and contains a simple run-and-evaluate script applying it to the infamous MNIST data set. Some of design choices I made (which you can also see in the code) are:

- tanh type activation functions for the hidden layers
- softmax type activation function for the output layers (since classification of digits is mutually exclusive)
- stochastic gradient descent using custom size batches (used for training)
- L^2 type regularization term (used for training)

The files:

- "mnist_data" folder:          Contains the mnist dataset split into x_train, y_train, and X_test
- "mnist_data.py":          Contains all the functions/object classes used to load and preprocess the data
                            NOTE: The class loading the mnist data is taken from Richard Marko's github repo "python-mnist".
                            I copy-pasted as appropriate to make this model self-contained.
- "TwoLP_classes.py":       Contains the neural network class "TwoLP" and all related functions. This is an object class that                                 
                                can 'survive' on its own and can be used for datasets and applications other than MNIST.
- "run_TwoLP_on_MNIST.py":  Script used for applying the neural net to the MNIST data. Usage is
                                 "python -m run_TwoLP_on_MNIST" from within the 'TwoLP' folder. Type
                                 "python -m run_TwoLP_on_MNIST -h" for information on model/script parameters.

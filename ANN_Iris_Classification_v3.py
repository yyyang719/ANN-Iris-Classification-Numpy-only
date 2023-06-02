#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 11:12:34 2022

@author: yuanyuan
"""
# the following coding includes four parts:
# 1. data processing function: deals with label data encoding, min-max normalization, data splitting, etc
# 2. dense layer class: applys the perceptron learning algorithm for each neuron in each layer
# 3. neural network class: construnct neural netowrk architecture, and applys the learning algorithms:
#    the forward propagation and the backward propagation
# 4. the main function: setting parameters, training nerual network, and print results

import numpy as np
import matplotlib.pyplot as plt
# set the random seed to have reproductive results,
np.random.seed(47)


def data_processing(values, labels, num_folds):
    # data processing will generate three kinds of data:
    # 1. train data: 80%
    # 2. test data: 20%
    # 3. k folds for train data: evenly partition train data into k folds for cross validation use
        
    # encode the labels into numerical value. cross-entropy loss is used later in
    # this multi-classification problem, no need to do one-hot encoding for the later calculation
    label_vector = np.zeros((len(labels),1)) 
    for i in range(len(labels)):
        if labels[i] == 'Iris-setosa':
            label_vector[i] = 0
        elif labels[i] =='Iris-versicolor':
            label_vector[i] = 1
        elif labels[i]== 'Iris-virginica':
            label_vector[i] = 2
    
    # normalize feature values: min-max normalization
    norm_value = (values - values.min(axis=0))/(values.max(axis=0) - values.min(axis=0))
            
    # combine normalized feature values and label vector to generate the Iris dataset
    data_set = np.append(norm_value, label_vector, axis=1)
    
    # randomlize dataset to mix up the order of data for three classes
    data_permu = np.random.permutation(data_set)
    
    # split data to train data (80%) and test date (20%)
    data_length=len(data_permu)
    train_length=int(0.8*data_length)
    
    train=data_permu[:train_length]
    test=data_permu[train_length:]
    
    # x is 2d array as input feature vectors(4 features), y is 1d output vector with value: 0, 1, 2
    x_train = train[:, :4]
    y_train = np.int32(train[:, 4:]).reshape(-1)  
    x_test = test[:, :4]
    y_test = np.int32(test[:, 4:]).reshape(-1)
    
    # evenly partition x train and y train into k folds for k-fold cross validation use,
    # each fold should be the same size, Every data item should appear in exactly one fold
    fold_size = int(len(x_train)/num_folds)
    x_folds = []
    y_folds = []

    for i in range(num_folds):
        x_per_fold = x_train[i*fold_size : (i+1)*fold_size, :]
        y_per_fold = y_train[i*fold_size : (i+1)*fold_size]
    
        x_folds.append(x_per_fold)
        y_folds.append(y_per_fold)
    
    return x_train, y_train, x_folds, y_folds, x_test, y_test


class Dense_Layer:
    # this class is about each neuron in every layer,
    # each neuron applys the perceptron learning algorithm
    # " d_ " and " d " in the following variables denote derivative
    
    def __init__(self, neurons):
        # set how many neurons in each layer
        self.neurons = neurons
        
    def relu(self, p):
        # using relu activation function for hidden layers, p is potential: the weighted sum
        return np.maximum(0, p)

    def softmax(self, p):
        # using softmax activation function for output layer, p is potential: the weighted sum
        exp_scores = np.exp(p)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs
    
    def relu_derivative(self, d_O, p):
        # for backward calculation
        d_p = np.array(d_O, copy = True)
        d_p[p <= 0] = 0
        return d_p
    
    def forward_calculation(self, inputs, weights, bias, activation):
        # calculate potential(the weighted sum) first, then go through the activation function
        # matrix dot product for all neurons in one layer
        potentials = np.dot(inputs, weights.T) + bias 
        
        if activation == 'relu':
            output = self.relu(potentials)
        elif activation == 'softmax':
            output = self.softmax(potentials)
            
        return output, potentials
    
    def backward_calculate(self, d_current_output, W_curr, potentials, previous_output, activation):
        # backward propagate the loss through the output layer
                
        if activation == 'softmax': # for the output layer
            # delta W = output * d_current_output (derivative)
            # learning rate will be applied in weights updata part
            d_W = np.dot(previous_output.T, d_current_output)
            d_b = np.sum(d_current_output, axis=0, keepdims=True)
            d_Output = np.dot(d_current_output, W_curr) 
        # for the hidden layers, we should calculate the chain derivative of 
        # its activation function first
        else:
            d_p = self.relu_derivative(d_current_output, potentials)
            d_W = np.dot(previous_output.T, d_p)
            d_b = np.sum(d_p, axis=0, keepdims=True)
            d_Output = np.dot(d_p, W_curr)
            
        return d_Output, d_W, d_b
    
       
class Neural_Network:
    # the main function for this part is:
    # construct neural network, randomly initialize weights between -0.1 and +0.1,
    # recursively do the forward propagation, the backward propagation, and weights update in training function
   
    def __init__(self):
        # initialize empty layers
        self.layers = [] 
        # initialize the whole network that includes layers and neurons in each layer
        self.architecture = [] 

        
    def add_layer(self, layer):
        self.layers.append(layer)
        
            
    def construct_neural_network(self, feature_size):
        # construct the architecture of the neural network
        # the input unites for Neural network equals the size of features in our Iris dataset
        for idx, layer in enumerate(self.layers):
            # input layer
            if idx == 0: 
                self.architecture.append({'input_dim':feature_size, 'output_dim':self.layers[idx].neurons,
                                         'activation':'relu'})
            # hidden layers
            elif idx > 0 and idx < len(self.layers)-1:
                self.architecture.append({'input_dim':self.layers[idx-1].neurons, 'output_dim':self.layers[idx].neurons,
                                         'activation':'relu'})
            # output layer
            else:
                self.architecture.append({'input_dim':self.layers[idx-1].neurons, 'output_dim':self.layers[idx].neurons,
                                         'activation':'softmax'})

    
    def init_weights(self):
        # randomly initialize weights matrix and bias for each layer
        # the dimention for each layer's weights matrix is 
        # np.array((the number of the neurons in this layer, the number of the inputs in this layer))
        # the dimention of bias in each layer is the same with the number of neurons in this layer
        self.weights_bias = [] # weights and bias for each layer, a list of dictionaries
      
        for i in range(len(self.architecture)): # for every layer
            self.weights_bias.append({
                'W':np.random.uniform(low=-0.1, high=0.1, # initialize weights between -0.1 and +0.1
                  size=(self.architecture[i]['output_dim'], 
                        self.architecture[i]['input_dim'])),
                'b':np.zeros((1, self.architecture[i]['output_dim']))})
        
    
    def forward_propagation(self, data):
        # feedforward fully connected neural network,
        # values are all represented as input in this part, 
        # the output of one layer actually is the input of its next layer.
        # potentials is the weighted sum prepared for its corresponding activation function
        self.memory = [] # store input and potentials for each layer
        current_input = data
        # go through each layer from the input layer
        for i in range(len(self.weights_bias)):
            previous_input = current_input
            current_input, potentials = self.layers[i].forward_calculation(inputs=previous_input, weights=self.weights_bias[i]['W'], 
                                           bias=self.weights_bias[i]['b'], activation=self.architecture[i]['activation'])
            
            # store input and potential(the weighted sum) of each layer for backward propagation use
            self.memory.append({'inputs':previous_input, 'P':potentials})
        
        # all data are represented as input, the final return of "current_input" actually
        # is the output of the last layer (output layer)
        return current_input

    
    def backward_propagation(self, predicted, actual):     
        # using the gradient-decent to updat weights and bias.
        # " d_ " and " d " in the following variables denote derivative
        num_samples = len(actual)
        
        # compute the gradient on predictions from the output layer,
        # derivative of cross-entropy loss function with respect to the output of the neural network is:
        # probability -1 (we use softmax activation function in the output layer)
        self.derivatives = [] # store derivative of weights and bias
        d_L = predicted
        # actual is the true label: 0, 1, or 2
        d_L[range(num_samples), actual] -= 1 # the predicted probability for the correct label minus 1
        d_L /= num_samples
        
        d_previous_output = d_L
        
        for idx, layer in reversed(list(enumerate(self.layers))): # backward
            # calculate and store from the output layer
            d_current_output = d_previous_output
            # values are all represented as output in this part
            previous_output = self.memory[idx]['inputs']
            potentials = self.memory[idx]['P'] 
            W_curr = self.weights_bias[idx]['W']
            
            activation = self.architecture[idx]['activation']

            d_previous_output, dW_curr, db_curr = layer.backward_calculate(d_current_output, W_curr, potentials, previous_output, activation)

            self.derivatives.append({'dW':dW_curr, 'db':db_curr}) # for weights and bias update use
            
    
    def cross_entropy_loss(self, predicted, actual):
        # multi classification problem uses cross-entropy loss here
        samples = len(actual)
        # categorical cross-entropy loss: the negative logarithm of 
        # the predicted probability for the correct class
        correct_logprobs = -np.log(predicted[range(samples),actual])
        # we use the average cross-entropy loss across all samples in the dataset
        loss = np.sum(correct_logprobs)/samples

        return loss


    def update_weights(self, lr):
        # lr is learning rate
        for idx, layer in enumerate(self.layers):
            self.weights_bias[idx]['W'] -= lr * list(reversed(self.derivatives))[idx]['dW'].T  
            self.weights_bias[idx]['b'] -= lr * list(reversed(self.derivatives))[idx]['db']
    

    def get_accuracy(self, predicted, actual):
        # find the index in the predicted results with highest probability, 
        # and compare them with the correct classes: 0, 1, or 2
        return np.mean(np.argmax(predicted, axis=1)==actual)
    
    
    def train_nn(self, x_train, y_train, epochs, lr):
        # recursively do the forward propagation, the backward propagation, and weights update
        
        self.loss = []
        self.accuracy = []
        # initialize weights between -0.1 to +0.1 
        self.init_weights()
        
        for i in range(epochs):
            # predict results with the forward propagation
            yhat = self.forward_propagation(x_train)
            
            # record accuracy and loss in the training process
            self.accuracy.append(self.get_accuracy(predicted=yhat, actual=y_train))
            self.loss.append(self.cross_entropy_loss(predicted=yhat, actual=y_train))
            
            # the loss is propagated backward through the gradient-descent algorithim
            self.backward_propagation(predicted=yhat, actual=y_train)   
            
            # update weights and biases with the learning rate after each backward propagation
            self.update_weights(lr)
 
    
    def kfold_crossvalidation(self, x_folds, y_folds, epochs, lr, k):
        # split train and validate data, use one of k folds as validate data, other (k-1) as train data
        train_data_length = len(x_folds[0])*(k-1)
        x_folds_arry = np.array(x_folds)
        y_folds_arry = np.array(y_folds)
        
        train_accu_avg = list() # initialize accu value
        validate_accu_avg = list()
        train_loss_avg = list() # initialize loss value
        validate_loss_avg = list()
        
        for epoch in range(1, epochs+1):
            
            train_accu_sum = 0
            validate_accu_sum = 0
            train_loss_sum = 0
            validate_loss_sum = 0

            for i_fold in range(k): # x_folds[0] ... x_folds[4]
                
                # generate train data, delete ith fold
                x_train = np.delete(x_folds_arry, i_fold, 0)
                x_train = np.reshape(x_train, (train_data_length, 4))
                y_train = np.delete(y_folds_arry, i_fold, 0)
                y_train = np.reshape(y_train, train_data_length)
                
                # extract ith fold as validate data
                x_validate = x_folds_arry[i_fold]
                y_validate = y_folds_arry[i_fold]
                
                # randomly initialize weights 
                self.init_weights()
                
                for i in range(epoch):
                    # training neural network
                    yhat = self.forward_propagation(x_train)                
                    self.backward_propagation(predicted=yhat, actual=y_train)            
                    self.update_weights(lr)
                
                # calculate accuracy and loss for train and validate data
                train_yhat = self.forward_propagation(x_train)
                train_accu = self.get_accuracy(predicted=train_yhat, actual=y_train)
                train_loss = self.cross_entropy_loss(predicted=train_yhat, actual=y_train)
                
                validate_yhat = self.forward_propagation(x_validate)
                validate_accu = self.get_accuracy(predicted=validate_yhat, actual=y_validate)
                validate_loss = self.cross_entropy_loss(predicted=validate_yhat, actual=y_validate)
                
                train_accu_sum += train_accu
                validate_accu_sum += validate_accu
                train_loss_sum += train_loss
                validate_loss_sum += validate_loss
            # k times train with one of folds as validate data, average the train and validate results
            train_accu_avg.append(train_accu_sum/k)
            validate_accu_avg.append(validate_accu_sum/k)
            train_loss_avg.append(train_loss_sum/k)
            validate_loss_avg.append(validate_loss_sum/k)

        return train_accu_avg, validate_accu_avg, train_loss_avg, validate_loss_avg
        

def main():
        
    # load Iris data as numpy array
    values = np.loadtxt('Iris_data.txt', delimiter=',', usecols=[0,1,2,3]) # load feature values
    labels = np.loadtxt('Iris_data.txt', dtype=object, delimiter=',', usecols=[4]) # load labels
    
    num_folds = 5 # k = 5 for our k-fold cross validation
    # generate train, validate, and test data
    x_train, y_train, x_folds, y_folds, x_test, y_test = data_processing(values, labels, num_folds)
    
    input_features = x_train.shape[1] # feature numbers for our input layer
    
    # create neural network object, add two hidden layers, each with 128 neurons, and output layer with 3 classes
    model = Neural_Network()
    model.add_layer(Dense_Layer(128)) 
    model.add_layer(Dense_Layer(128))
    model.add_layer(Dense_Layer(3))
    
    # construct the whole neural network architecture:
    # input layer: with size of input features
    # two hidden layers: with size 128
    # output layer: 3 classes output
    model.construct_neural_network(input_features)
    
    learning_rate = 0.1 # for weights updating
    epochs = 800 # this number is checked by the k-fold cross validation to avoid overfitting
    
    # # k-fold cross validation to check the overfitting situation (find the proper epoch number)
    # train_accu_avg, validate_accu_avg, train_loss_avg, validate_loss_avg = model.kfold_crossvalidation(x_folds, y_folds, epochs, learning_rate, num_folds)
    
    model.train_nn(x_train, y_train, epochs, learning_rate) # training neural network
    
    # calculate the accuracy and loss for the test data
    test_accuracy = model.get_accuracy(model.forward_propagation(x_test), y_test)    
    test_loss = model.cross_entropy_loss(model.forward_propagation(x_test), y_test)
    
    # print the train results for the last epoch
    print('EPOCH: {}, TRAIN ACCURACY: {}, TRAIN LOSS: {}'.format(epochs, np.round(model.accuracy[-1], 3), np.round(model.loss[-1],3)))
    # print the test results              
    print("TEST ACCURACY: {}".format(np.round(test_accuracy, 3)))  
    print("TEST LOSS: {}".format(np.round(test_loss, 3)))
    
    # plot all training results generated from the training process 
    epoch_list = [i for i in range(epochs)]
    plt.plot(epoch_list, model.accuracy, color = 'red', label = "train accuracy")
    plt.plot(epoch_list, model.loss, color = 'blue', label = "train loss")
    plt.title('accuracy and loss for training data')
    plt.xlabel('epochs')
    plt.ylabel('performance')
    plt.legend()
    plt.show()  


if __name__ == "__main__":
    main()

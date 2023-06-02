# ANN-Iris-Classification-Numpy-only
data processing:
1. encode the labels of Iris data into numerical value: 0, 1, 2. cross-entropy loss function is used later
   in this multi-classification problem, no need to do one-hot encoding for the later calculation.
2. min-max normalization for feature values of the Iris data, and the order of dataset will be randomly mixed up.
3. test data: 20%, train data: 80%, meanwhile, evenly partition train data into k(=5) folds for cross validation use.

architecture of the neural network model:
inter-layer connectivity policy: fully connected nerual network.
1. input layer: with size of input features (=4).
2. two hidden layers: with size 128 neurons.
3. output layer: 3 classes output

structure of neurons:
learning rule: delta learning rule
1. neurons in the output layer: with softmax activation function.
2. neurons in all other layers: with Relu activation function.

Parameters of the model:
1. weights are randomly initialized between -0.1 and +0.1
2. loss function: cross-entropy loss function for our multi classificaiton problem.
   The cross-entropy loss equals the negative logarithm of the predicted probability for the correct class.
   we uses the average cross-entropy loss across all samples in the dataset as our final loss.
3. learning rate to update weights: 0.1. different learning rates have been tried, this one generates smooth decreasing loss curve.

learning algorithm to train neural network:
1. the forward propagation.
2. the backward propagation.

measurement of performance:
1. k-fold cross validation to make sure that we set proper number of epoch to train the model without overfitting.
2. the accuracy and loss for the train data and test data. 
3. final results for test data: the test accuracy is 1, and the test loss is 0.054.

two images are saved for measurement of performance:
the first one is k-fold cross validation results, which shows that there is no overfitting during the whole training process.
the second one is the results for train data through the whole training process.

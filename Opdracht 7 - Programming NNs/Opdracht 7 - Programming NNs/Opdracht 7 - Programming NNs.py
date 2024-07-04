import numpy as np
import random

def activation_sigmoid(input):
    """Calculates the sigmoid activation function of input."""
    return 1 / (1 + np.exp(-input)) #https://www.codecademy.com/resources/docs/numpy/built-in-functions/dot

def derivative_sigmoid(input):
    """Calculates the derivative (gradient) of the sigmoid activation function of an input."""
    return 1 / (1 + np.exp(-input)) * (1 - 1 / (1 + np.exp(-input))) #https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e

def import_dataset(file):
    """Imports dataset"""
    data = np.genfromtxt(file, delimiter=",", usecols=[0, 1, 2, 3])
    return data

def import_labels(file):
    """Imports the labels"""
    labels = np.genfromtxt(file, delimiter=",", usecols=[4], dtype=str)
    return labels

def onehot_encoding_labels(labels): #https://stackoverflow.com/questions/40207422/binary-numbers-instead-of-one-hot-vectors
    """Performs one-hot encoding for a list of labels based on the Iris dataset classes."""
    outputs =[]
    for label in labels:
        if label == "Iris-setosa":
            outputs.append([1, 0, 0])
        elif label == "Iris-versicolor":
            outputs.append([0, 1, 0])
        elif label == "Iris-virginica":
            outputs.append([0, 0, 1])
    return outputs

def normalize_data(data):
    """Normalizes the dataset"""
    min = np.min(data)
    max = np.max(data)

    normalizedData = (data - min) / (max - min) ##https://rayobyte.com/blog/how-to-normalize-data-in-python/
    return normalizedData

def split_dataset(array, percentage): #https://stackoverflow.com/questions/49054538/how-to-split-the-data-set-without-train-test-split
    """Splits the dataset into a training set and test set based on a certain percentage"""
    train_percentage_index = int(percentage * len(array))
    train_data, test_data = array[:train_percentage_index], array[train_percentage_index:]
    return train_data, test_data
    
class Neuron:
    """Initializes a neural network layer with random weights and bias.
        Variables:
        input_data : list
            List to store input data for this layer.
        weights : list
            List to store weights connecting this layer to the previous layer.
        output : float
            Output value of this layer.
        error : float
            Error value for this layer.
        actual_output : float
            Actual output value of this layer.
        previous_layer : int
            Number of nodes or neurons in the previous layer.
        bias : float
            Bias value for this layer."""
    def __init__(self, previous_layer):
        self.input_data = []
        self.weights = []
        self.output = 0
        self.error = 0
        self.actual_output = 0
        self.previous_layer = previous_layer

        for _ in range(self.previous_layer):
            self.weights.append(random.uniform(0,1))

        self.bias = random.uniform(0,1)
    
    def feed_forward(self, input):
        """Performs the feedforward computation for this neural network layer.
        - Updates the `input_data` variable with the input data.
        - Calculates the dot of the input and weights plus the bias.
        - Applies the sigmoid activation function (activation_sigmoid) to self.input to get self.actual_output.
        - Returns the output of the activation function."""
        self.input_data = input
        self.input = np.dot(input, self.weights) + self.bias 
        self.actual_output =  activation_sigmoid(self.input)
        return self.actual_output
    
    def update_weights(self, learning_rate):
        """Updates the weights.
        - Uses the error of the layer to adjust weights and bias.
        - Updates each weight based on the input data and the error.
        - Updates the bias based on the error and learning rate."""
        for index, _ in enumerate(self.input_data):
            self.weights[index] += (learning_rate * self.error * self.input_data[index]) #New Weight = Old Weight + (Learning Rate * Left Neuron * △Right Neuron)

        self.bias += learning_rate * self.error

class neural_network:
    """Initializes a neural network with specified sizes for input, hidden, and output layers.
        Variables:
        learning_rate : float
            Learning rate for weight updates during training.
        size_input_layer : int
            Number of neurons in the input layer.
        size_hidden_layer : int
            Number of neurons in the hidden layer.
        size_output_layer : int
            Number of neurons in the output layer.
        input_layer : list of Neuron objects
            List containing Neuron objects representing the input layer.
        hidden_layer : list of Neuron objects
            List containing Neuron objects representing the hidden layer.
        output_layer : list of Neuron objects
            List containing Neuron objects representing the output layer."""
    def __init__(self, size_input_layer, size_hidden_layer, size_output_layer, learning_rate):
        self.learning_rate = learning_rate
        self.size_input_layer = size_input_layer
        self.size_hidden_layer = size_hidden_layer
        self.size_output_layer = size_output_layer

        self.input_layer = [Neuron(0) for _ in range(size_input_layer)]
        self.hidden_layer = [Neuron(size_input_layer) for _ in range(size_hidden_layer)]
        self.output_layer = [Neuron(size_hidden_layer) for _ in range(size_output_layer)]

    def estimate_outputs(self,input_data): 
        """Estimates the outputs of the neural network for given input data.
        - Performs feedforward propagation through the hidden layer and then through the output layer.
        - Returns the estimated outputs from the output layer neurons."""
        estimated_outputs = []
        
        hidden_outputs = [neuron.feed_forward(input_data) for neuron in self.hidden_layer]
   
        estimated_outputs = [neuron.feed_forward(hidden_outputs) for neuron in self.output_layer]

        return estimated_outputs

    def call_update_weights(self):
        """Calls the update_weights method for all neurons in the output and hidden layers."""
        for neuron in self.output_layer:
            neuron.update_weights(self.learning_rate)
        for neuron in self.hidden_layer:
            neuron.update_weights(self.learning_rate)

    def backpropagation(self, desired_output):
        """Performs backpropagation to calculate errors for output and hidden layer neurons.
        - Calculates errors for output layer neurons based on the difference between desired and actual outputs.
        - Propagates these errors backward to calculate errors for hidden layer neurons."""
        for index, output_neuron in enumerate(self.output_layer):
            output_neuron.error = derivative_sigmoid(output_neuron.actual_output) * (desired_output[index] - output_neuron.actual_output) #△output neuron = Afgeleide van Output Neuron * (Desired output - Actual Output)

        for index, hidden_neuron in enumerate(self.hidden_layer):
            hidden_neuron.error = derivative_sigmoid(hidden_neuron.actual_output) * sum(output_neuron.weights[index] * output_neuron.error for output_neuron in self.output_layer) #△Hidden Neuron(en) = Afgeleide Hidden Neuron * (Weight * △Output Neuron)

    def train(self, input_dataset, desired_output, epochs):
        """Trains the neural network using input dataset and desired outputs over multiple epochs.
        - Executes forward propagation, backward propagation, and weight updates for each input and output pair in the dataset,
            for the specified number of epochs (100 in the tests)."""
        for _ in range(epochs):
            for input_data, output_data in zip(input_dataset, desired_output):
                self.estimate_outputs(input_data)
                self.backpropagation(output_data)
                self.call_update_weights()

def one_hot_comparing(estimation ,actual):
    """Compares a one-hot encoded estimation with an actual one-hot encoded label.
    - Converts the estimation to a one-hot format based on the maximum value.
    - Compares the converted estimation with the actual one-hot encoded label."""
    one_hot_index = estimation.index(max(estimation)) #When using one-hot encoding the max estimated chance is being used, because it is the most accurate estimation.
    for index, _ in enumerate(estimation):
        if index == one_hot_index:
            estimation[index] = 1
        else:
            estimation[index] = 0
    return estimation == actual

def testing_NN(NN, test_data, test_outpus):
    """Tests the neural network on test data and evaluates its performance.
    - Uses the estimate_outputs method of the neural network to predict outputs for each test data point.
    - Compares predicted outputs with expected outputs using one_hot_comparing function.
    - Counts and returns the number of correctly estimated outputs."""
    results = []
    correct_estimations = 0
    for i, values in enumerate(test_data):
        estimated_outputs = NN.estimate_outputs(values)
        results.append(one_hot_comparing(estimated_outputs, test_outpus[i]))
    
    for result in results:
        if result == True: 
            correct_estimations += 1

    return round(correct_estimations)
 
if __name__ == "__main__":
    """This is the main :p"""
    np.random.seed(0)

    data = import_dataset("Dataset/iris.data")
    labels = import_labels("Dataset/iris.data")
    normalized_data    = normalize_data(data)
    binary_encoded_labels = onehot_encoding_labels(labels)

    train_data, test_data     = split_dataset(normalized_data, 0.82)
    train_labels, test_labels = split_dataset(binary_encoded_labels, 0.82)

    Neural_network = neural_network(4, 8, 3, 0.1)
    Neural_network.train(train_data, train_labels, 1000) #if you do a higher epoch, it takes some time to calculate so have patience :)

    correct_estimations = testing_NN(Neural_network ,test_data ,test_labels)

    print("Test results with test data: ")
    print("Test results: correct_estimations is: " + str(correct_estimations) + " and the precentage is: " + str(correct_estimations/len(test_data) * 100) + "%")
    
    print("------------------------------------------------------------------------")
    correct_estimations = testing_NN(Neural_network ,normalized_data ,binary_encoded_labels)
    print("Test results with whole dataset")
    print("Test results: correct_estimations is: " + str(correct_estimations) + " and the precentage is: " + str(correct_estimations/len(normalized_data) * 100) + "%")
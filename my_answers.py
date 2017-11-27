import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        ## activation_function of the hidden layer is the sigmoid function ####
        self.activation_function = lambda x : 1 / (1+np.exp(-x))  # Replace 0 with your sigmoid calculation.

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):

            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):

        def forward_pass(raw_input, weights, activation_function):
            inputs = np.dot(raw_input, weights)
            outputs = activation_function(inputs)
            return outputs

        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        ### Forward pass ###
        hidden_outputs = forward_pass(X, self.weights_input_to_hidden, self.activation_function)
        final_outputs = forward_pass( hidden_outputs, self.weights_hidden_to_output, lambda x : x )

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        # Backpropagated error terms
        output_grad = 1
        output_error_term = error * output_grad # whereas 1 is the derivative of the activation function f(x)=x
        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs[:,None]

        # Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)

        # derivative of the activation function applied to the input of the layer
        # sigmoid_prime(x) = sigmoid(x) * (1-sigmoid(x)) with
        # hidden_output = sigmoid(x)
        # --> sigmoid_prime(x) = hidden_output * (1-hidden_output)
        hidden_grad = hidden_outputs * (1-hidden_outputs)
        hidden_error_term = hidden_error * hidden_grad
        
        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:,None]

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        # updates the weight with the calculated delta divided by the number of training records
        # and multiplied by the learning rate
        self.weights_hidden_to_output += self.lr * delta_weights_h_o/n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h/n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
		# The forward pass for the execution and the forward pass for the training is actually identical,
        # so that the training method can be reused.
        final_outputs, hidden_outputs = self.forward_pass_train(features)

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 1000
learning_rate = .1
hidden_nodes = 40
output_nodes = 1

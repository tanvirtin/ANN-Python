from random import random, seed
from math import exp
from Neuron import Neuron

seed(1) # seeding the random time to get similar random weights every time

# in this code ith index will always be used to iterate over layers in the Neural Network
# in this code jth index will always be used to iterate over neurons in a layer of a Neural Network
# index k will be used to iterate over weights list of a neuron in an neural network


class NeuralNetwork(object):

	def __init__(self, dimension, learning_rate):
		self.layers = self.__generate_layers(dimension)
		self.alpha = learning_rate

	'''
		Purpose: Double underscore method called __repr__ to return a string
				 which for a better representation of the Neural Network when printed.
	'''
	def __repr__(self):
		string = ""
		for i in range(len(self.layers)):
			string += "Layer: " + str(i + 1) + "\n"
			string += "-------\n"
			for j in range(len(self.layers[i])):
				string += "----------------\n"
				string += "|||| Neuron ||||\n"
				string += "----------------\n"
				string += str(self.layers[i][j])
				string += "\n"
			string += "\n"
		return string

	'''
		Purpose: Generates the layers of a Neural Network with the dimensions provided
				 through the constructor. Architecture of a neuron in a hidden layer is
				 as follows, the hidden layer will contain weights in individual neurons
				 which is a list and the length of the list is equal to the number of neurons
				 in the previous layer. This means that the weight output from the previous
				 layer gets multiplied with the weight in the current layers neuron, and the
				 index of the weight array to grab the weight from in the current neuron, is
				 equal to the index of the neuron in the previous layer. We also need to generate
				 an additional weight for each neuron in the current layer known as the bias 
				 neuron, that's why we do len(self.layers[i - 1]) + 1) at line 47
	'''
	def __generate_layers(self, dimension):
		if isinstance(dimension, list):
			self.layers = []
			for i in range(len(dimension)):
				list_of_neurons = [] # list of neurons in each layer
				for j in range(dimension[i]): # dimension array contains how many neurons should be in each layer
					if i - 1 > -1: # to stop going below 0th index, first layer should not have neurons with any weights
						# NOTE** - we are adding + 1 to the length of the neurons in the previous layer because we want an additional weight value which will be the bias
						list_of_neurons.append(Neuron(len(self.layers[i - 1]) + 1, random)) # each neuron will contain a n number of weights in a list where n is the length of the neurons in the previous layer
					
					else: # first layer, also known as input layer with no weights
						list_of_neurons.append(Neuron(0, random))
			
				self.layers.append(list_of_neurons) # makes a single list containing several lists of neurons representing a layer

			return self.layers

		else:
			raise Exception("Invalid dimension type")
		
	'''
		Purpose: A list of input is taken and is feed forwarded through the Neural Network.
				 The input values follow sigma->weight-i->j * input-i rule and then the neuron is
				 activated using an activation function
	'''
	def __feed_forward(self, input_list):
		if not isinstance(input_list, list) or len(input_list) != len(self.layers[0]): # either of the statement needs to be true for the entire expression to evaluate to true
			raise Exception("Invalid input list")


		# loop over the neurons of the Neural Network and assign the elements of the input list to the neurons output
		for j in range(len(self.layers[0])):
			self.layers[0][j].output = input_list[j]


		# loops over the hidden layers of a Neural Network to the output layer and feeds the input  
		for i in range(1, len(self.layers)): # loops through layers

			for j in range(len(self.layers[i])): # loops through neurons in a layer

				for k in range(len(self.layers[i - 1])): # loops through neurons in the previous layer
					# NOTE** - index of neurons at the previous layer k is equal to the index of the weights in the current neuron and each weight is connected to only a single neuron!
					self.layers[i][j].output += self.layers[i - 1][k].output * self.layers[i][j].weights[k]

				# the last for loop did not add the bias term as for example in the previous layer
				# we have 2 neurons and our weight is 3, we loop according to the previous layer, the
				# last bias term never gets accessed with k as k loops using the length of the previous layer neurons

				# bias weights always gets multiplied by 1
				self.layers[i][j].output += self.layers[i][j].weights[len(self.layers[i][j].weights) - 1] * 1

				# we don't sigmoid the last layer
				if i != len(self.layers) - 1:
					# individual neuron is being activated upon exceeding the threshold value
					self.layers[i][j].output = self.__sigmoid(self.layers[i][j].output)

	'''
		Purpose: Method of the class entirely responsible for the learning process of a
				 Neural Network
	'''
	def __back_propagate(self, target_list):
		if not isinstance(target_list, list) or len(target_list) != len(self.layers[len(self.layers) - 1]):
			raise Exception("Invalid target list")

		# loops over each neuron in the output layer and applies the following formula
		# delta_e(theta) = (t - 0) * sigmoidPrime(output(theta))
		for j in range(len(self.layers[len(self.layers) - 1])):
			self.layers[len(self.layers) - 1][j].delta_e = (target_list[j] - self.layers[len(self.layers) - 1][j].output) * self.__sigmoid_prime(self.layers[len(self.layers) - 1][j].output)


		# now we loop all the hidden layers starting backwards not including the input layer and calculate
		# the delta_e error values for the neurons in the hidden layer
		# delta_e(j) = sigma->error-k * weight-k->j

		for i in range(len(self.layers) - 2, 0, -1): # loops through each hidden layer in a Neural Network starting from the back
			for j in range(len(self.layers[i])): # loops through neurons in a layer

				# jth index for the neuronin the current layer has a weight which links itself
				# to the i + 1 layer in the next layer has the same index j, as one weight has only
				# one neuron connecting it to it

				for k in range(len(self.layers[i + 1])): # loops through neurons in the next layer
					self.layers[i][j].delta_e += self.layers[i + 1][k].weights[j] * self.layers[i + 1][k].delta_e

				# delta_e(j) = (delta_e(k)) * sigmoidPrime(output(j))
				self.layers[i][j].delta_e = self.layers[i][j].delta_e * self.__sigmoid_prime(self.layers[i][j].output)

		self.__stochastic_gradient_descent()

	'''
		Purpose: Loops over all the neurons in every layer except the input layer
				 and applies the changes to the weights delta_e which are stored as in
				 as delta_e attribute. This is the moment we have been waiting for
				 dEj/dWi->j =  delta(j) * output-i, this is the formula to find the
				 gradient and now we have all the pieces, we take the change in error value
				 multiply times some learning rate and times with the output of the neuron in the
				 previous layer for which the weight belongs to and update the weight.

				 NOTE** - multiplying the delta error value with the output of the neuron in the previous
				 		  layer gives us the gradient of the cost function, also known as the rate of 
				 		  change of error in a Neural Network with respect to the weight of an individual
				 		  layer

				 weightj += alpha * delta_e * outputi
				 => weightj += alpha * dEj/dWi->j

				 We descend down the gradient at a rate and not all at once hence the name gradient
				 descent, we descend down the gradient using the stochastic gradient descent meaning
				 each change in weight value updates the weight immedietly in each epoch.
					
	'''
	def __stochastic_gradient_descent(self):
		# starts looping from the hidden layers and skips the input layer as it contains no weights, hence no weights need to be updated
		for i in range(1, len(self.layers)): # loops over the layers of a Neural Network

			for j in range(len(self.layers[i])): # loops over each neurons in a layer

																	# loops over each weight in a neuron
				for k in range(len(self.layers[i][j].weights) - 1): # -1 is because we don't want to include the bias weight here
					# the weight that I contain times the input with with I multiply myself to get the error in the first place
					self.layers[i][j].weights[k] += self.alpha * self.layers[i][j].delta_e * self.layers[i - 1][k].output # again self.layers[i - 1][k].output, k because each weight is destined to have only one input 

				# bias weight always gets multiplied by 1, because the input in the previous layer is still 1, so therefore we multiply it using 1 seperately
				self.layers[i][j].weights[len(self.layers[i][j].weights) - 1] += self.alpha * self.layers[i][j].delta_e * 1 # 1 because thats what the bias weight gets multiplied by

	'''
		Purpose: A simple method thats provides abstraction for training the Neural Network by using private methods
	'''
	def train(self, input_list, output_list):
		self.__feed_forward(input_list)
		self.__back_propagate(output_list)

	'''
		Purpose: Provides abstraction in querying the Neural Network by combining private methods
	'''
	def query(self, input_list):
		self.__feed_forward(input_list)
		out = []
		for j in range(len(self.layers[len(self.layers) - 1])):
			out.append(self.layers[len(self.layers) - 1][j].output)
		return out

	'''
		Purpose: Returns the outputs of the Neural Network in a string
	'''
	def output_to_string():
		string = "------\n"
		# loops over the last layer of the Neural Network which is the output layer
		for j in range(len(self.layers[len(self.layers) - 1])):
			string += "Output --> " + str(self.layers[len(self.layers) - 1][j].output) + "\n"
		string += "------\n"

	'''
		Purpose: one of many activation function sigmoid, what sigmoid does is, it takes in a 
				 value and squashes the value within a range of 0 and 1, this activation function
				 is used when our threshold value is exceeded
	'''
	def __sigmoid(self, x):
		return 1.0 / (1.0 + exp(-x))

	'''
		Purpose: derivative of the sigmoid prime function with respect to the output (x)
	'''
	def __sigmoid_prime(self, x):
		return x * (1.0 - x)
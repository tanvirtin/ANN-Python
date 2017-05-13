from random import random, seed
from math import exp
from Neuron import Neuron

seed(1) # seeding the random time to get similar random weights every time

class NeuralNetwork(object):

	def __init__(self, dimension):
		self.layers = self.generate_layers(dimension)

	'''
		Purpose: Double underscore method called __repr__ to return a string
				 which for a better representation of the Neural Network when printed.
	'''
	def __repr__(self):
		string = ""
		for i in range(len(self.layers)):
			string += "Layer" + str(i + 1) + "\n"
			string += "-------\n"
			for j in range(len(self.layers[i])):
				string += "||||Neuron||||\n"
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
	def generate_layers(self, dimension):
		if isinstance(dimension, list):
			self.layers = []
			for i in range(len(dimension)):
				list_of_neurons = [] # list of neurons in each layer
				for j in range(dimension[i]): # dimension array contains how many neurons should be in each layer
					if i - 1 > -1: # to stop going below 0th index, first layer should not have neurons with any weights
						list_of_neurons.append(Neuron(len(self.layers[i - 1]) + 1, random)) # each neuron will contain a n number of weights in a list where n is the length of the neurons in the previous layer
					
					else: # first layer, also known as input layer with no weights
						list_of_neurons.append(Neuron(0, random))
			
				self.layers.append(list_of_neurons) # makes a single list containing several lists of neurons representing a layer

			return self.layers

		else:
			raise Exception("Invalid dimension type")
		
	'''
		Purpose: A list of input is taken and is feed forwarded through the Neural Network.
				 The input values follow sigma weight * input rule and then the neuron is
				 activated using an activation function
	'''
	def feed_forward(self, input_list):
		if not isinstance(input_list, list) or len(input_list) != len(self.layers[0]): # either of the statement needs to be true for the entire expression to evaluate to true
			raise Exception("Invalid input list")


		# loop over the neurons of the Neural Network and assign the elements of the input list to the neurons output
		for j in range(len(self.layers[0])):
			self.layers[0][j].output = input_list[j]


		# loops over the hidden layers of a Neural Network to the output layer and feeds the input  
		for i in range(1, len(self.layers)): # loops through layers

			for j in range(len(self.layers[i])): # loops through neurons in a layer

				for k in range(len(self.layers[i - 1])): # loops through neurons in the previous layer
					# NOTE** - index of neurons at the previous layer k is equal to the index of the weights in the current neuron
					self.layers[i][j].output += self.layers[i - 1][k].output * self.layers[i][j].weights[k]

				# the last for loop did not add the bias term as for example in the previous layer
				# we have 2 neurons and our weight is 3, we loop according to the previous layer, the
				# last bias term never gets accessed with k as k loops using the length of the previous layer neurons

				self.layers[i][j].output += self.layers[i][j].weights[len(self.layers[i][j].weights) - 1]

				# individual neuron is being activated upon exceeding the threshold value
				self.layers[i][j].output = self.__sigmoid(self.layers[i][j].output)

	'''
		Purpose: one of many activation function sigmoid, what sigmoid does is, it takes in a 
				 value and squashes the value within a range of 0 and 1, this activation function
				 is used when our threshold value is exceeded
	'''
	def __sigmoid(self, x):
		return 1.0 / (1.0 + exp(-x))
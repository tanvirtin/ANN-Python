from random import random, seed
from Neuron import Neuron

# seeding the random time to get similar random weights every time
seed(1)

class NeuralNetwork(object):
	def __init__(self, dimension):
		self.layers = self.generateLayers(dimension)


	def __repr__(self):
		string = ""

		for i in range(len(self.layers)):
			pass



	def generateLayers(self, dimension):
		if isinstance(dimension, list):
			self.layers = []
			for i in range(len(dimension)):

				for j in range(dimension[i]): # dimension array contains how many neurons should be in each layer
					
					list_of_neurons = [] # list of neurons in each layer
					
					if i - 1 > -1: # to stop going below 0th index, first layer should not have neurons with any weights
						list_of_neurons.append(Neuron(len(self.layers[i - 1]), random))
					
					else: # first layer, also known as input layer with no weights
						list_of_neurons.append(Neuron(0, random))
			
				self.layers.append(list_of_neurons) # makes a single list containing several lists of neurons representing a layer

			return self.layers

		else:
			raise Exception("Invalid dimension type")
		
from random import random, seed
from NeuralNetwork import NeuralNetwork
from Neuron import Neuron

seed(1)

def main():
	nn = NeuralNetwork([1, 2, 2, 1])

	print(nn)

if __name__ == "__main__":
	main()
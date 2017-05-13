from random import random, seed
from tqdm import tqdm
from NeuralNetwork import NeuralNetwork
from Neuron import Neuron

seed(1)

def main():
	nn = NeuralNetwork([1, 4, 4, 1], 0.5)

	epochs = 80000

	for i in tqdm(range(epochs)):
		nn.train([7], [1])
		nn.train([10], [0])

	print(nn.query([10]))

	print(nn.query([7]))

if __name__ == "__main__":
	main()
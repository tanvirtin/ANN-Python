from random import random, seed
from tqdm import tqdm
from NeuralNetwork import NeuralNetwork
from Neuron import Neuron

seed(1)

def main():
	nn = NeuralNetwork([2, 4, 4, 1], 0.5)

	epochs = 80000

	for i in tqdm(range(10000)):
		nn.train([0, 1], [0])
		nn.train([0, 0], [0])
		nn.train([1, 1], [1])


	print(nn.query([1, 1]))
	print(nn.query([0, 0]))
	print(nn.query([0, 1]))

if __name__ == "__main__":
	main()
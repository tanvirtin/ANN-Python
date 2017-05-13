
class Neuron(object):

	def __init__(self, num_weights, rand_func):
		self.weights = self.gen_weights(num_weights, rand_func)
		self.delta_e = 0
		self.output = 0

	def __repr__(self):
		string = "weights -> "
		string += str(self.weights) + "\n"
		string += "delta_e -> " + str(self.delta_e) + "\n"
		string += "output -> " + str(self.output) + "\n"
		return string


	def gen_weights(self, num_weights, rand_func):
		self.weights = []
		for i in range(num_weights):	
			self.weights.append(rand_func())

		return self.weights

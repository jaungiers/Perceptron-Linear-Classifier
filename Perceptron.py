import random
class Perceptron:
	
	def __init__(self, input_count):
		self.weights = []
		self.k = 0.01
		for n in xrange(input_count):
			self.weights.append(random.uniform(-1, 1))

	def FeedForward(self, inputs):
		sigma = 0
		for i, e in enumerate(inputs):
			sigma += e * self.weights[i]
		return self.Activate(sigma)

	def Activate(self, sigma):
		if sigma > 0:
			return 1
		else:
			return -1

	def Train(self, inputs, desired):
		guess = self.FeedForward(inputs)
		error_delta = desired - guess
		for i, w in enumerate(self.weights):
			self.weights[i] += error_delta * inputs[i] * self.k
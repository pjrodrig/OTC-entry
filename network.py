import random
import numpy as np

from utils import *

class Network:
    def __init__(self, layers, config=None):
        self.layers = layers
        self.init_weights_and_biases(config)

    def init_weights_and_biases(self, config=None):
        if(config):
            self.loadWeightsAndBiases(config)
        else:
            # pick random bias for each neuron per layer and store as an array for easy array addition
            self.biases = np.array([np.random.randn(layer, 1) for layer in self.layers[1:]])
            # pick random weights for each input to each neuron in each layer
            self.weights = np.array([np.random.randn(nextLayer, prevLayer) for prevLayer, nextLayer in zip(self.layers[:-1], self.layers[1:])])

    def load_weights_and_biases(self, config):
        self.weights = []
        self.biases = []

    def evaluate(self, values):
        activations = [values]
        connections = []
        for weights, bias in zip(self.weights, self.biases):
            dot = np.dot(weights, values) + bias
            connections.append(dot)
            values = sigmoid(np.array(dot))
            activations.append(np.array(values))
        return values, activations, connections

    def backPropagate(self):
        pass
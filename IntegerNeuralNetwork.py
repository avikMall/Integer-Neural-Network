import numpy as np


class NeuralNetwork:

    def __init__(self):
        np.random.seed(1)

        self.synapticWeights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidDerivative(self, x):
        return x * (1 - x)

    def train(self, trainingInputs, trainingOutputs, trainingIterations):
        for iteration in range(trainingIterations):
            output = self.think(trainingInputs)
            error = trainingOutputs - output
            adjustments = np.dot(trainingInputs.T, error * self.sigmoidDerivative(output))
            self.synapticWeights = self.synapticWeights + adjustments

    def think(self, inputs):

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synapticWeights))
        return output

    def training(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synapticWeights))
        if np.dot(inputs, self.synapticWeights) == 0:
            output = 0
        return output


if __name__ == "__main__":
    neuralNetwork = NeuralNetwork()

    print('Random synaptic weights: ')
    print(neuralNetwork.synapticWeights)

    trainingInputs = np.array([[0, 0, 1],
                               [1, 1, 1],
                               [1, 0, 1],
                               [0, 1, 1]])
    trainingOutputs = np.array([[0, 1, 1, 0]]).T

    neuralNetwork.train(trainingInputs, trainingOutputs, 20000)

    print('Synaptic weights after training: ')
    print(neuralNetwork.synapticWeights)

    A = str(input('Input 1: '))
    B = str(input('Input 2: '))
    C = str(input('Input 3: '))

    print('New situation: input data = ', A, B, C)
    print('Output data: ')
    print(neuralNetwork.training(np.array([A, B, C])))

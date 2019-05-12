import math
import ArchiveManager as aManager
import numpy as np
from builtins import range
from random import seed
from random import random


class Backpropagation:

    # Initilialize neural network with layers making a amount neurons
    def initialize(self, nInputs, nHidden, nOutputs):
        network = list()
        hiddenLayer = [{'weights': [random() for i in range(nInputs + 1)]} for i in range(nHidden)]
        network.append(hiddenLayer)
        outputLayer = [{'weights': [random() for i in range(nHidden + 1)]} for i in range(nOutputs)]
        network.append(outputLayer)
        return network

    # Propagate forward
    def activate(self, inputs, weights):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation

    def transfer(self, activation):
        return 1.0 / (1.0 + math.exp(-activation))

    def forwardPropagate(self, network, row):
        inputs = row
        for layer in network:
            newInputs = []
            for neuron in layer:
                activation = self.activate(inputs, neuron['weights'])
                neuron['output'] = self.transfer(activation)
                newInputs.append(neuron['output'])
            inputs = newInputs
        return inputs

    # Propagate backwards
    def transferDerivative(self, output):
        return output * (1.0 - output)

    def backwardPropagateError(self, network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if (i != len(network) - 1):
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.transferDerivative(neuron['output'])

    # For train network
    def updateWeights(self, network, row, learningRate, nOutputs):
        nOutputs = nOutputs * -1
        for i in range(len(network)):
            inputs = row[:nOutputs]
            if (i != 0):
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(network[i])):
                    neuron['weights'][j] += learningRate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += learningRate * neuron['delta']

    def updateLearningRate(self, learningRate, decay, epoch):
        return learningRate * 1 / (1 + decay * epoch)

    def trainingNetwork(self, network, train, learningRate, nEpochs, nOutputs, expectedError):
        sumError = 10000.0
        for epoch in range(nEpochs):
            if (sumError <= expectedError):
                break
            if(epoch % 100 == 0):
                learningRate = self.updateLearningRate(learningRate, learningRate/nEpochs, float(epoch))

            sumError = 0
            for row in train:
                outputs = self.forwardPropagate(network, row)
                expected = self.getExpected(row, nOutputs)
                sumError += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                self.backwardPropagateError(network, expected)
                self.updateWeights(network, row, learningRate, nOutputs)
            print('> epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learningRate, sumError))

    def getExpected(self, row, nOutputs):
        expected = []
        for i in range(nOutputs):
            temp = (nOutputs - i) * - 1
            expected.append(row[temp])
        return expected

    # For predict result
    def predict(self, network, row):
        outputs = self.forwardPropagate(network, row)
        return outputs


trainingSet = aManager.readtable('archives/training.dat')
trainingSet = np.matrix(trainingSet)
trainingSet = np.asfarray(trainingSet, int)

testSet = aManager.readtable('archives/test.dat')
testSet = np.matrix(testSet)
testSet = np.asfarray(testSet, int)

print('######################### MACHINE LEARNING WORK - BACKPROPAGATION #########################')
print('Student: Yury Alencar Lima')
print('Registration: 161150703\n')

nOutputs = int(input('Insert the number Neurons into Output Layer: '))
nEpochs = int(input('Insert the number of Epochs: '))
nHiddenLayer = int(input('Insert the number Neurons into Hidden Layer: '))
learningRate = float(input('Insert Learning Rate: '))
expectedError = float(input('Insert Expected Error: '))

seed(1)
backpropagation = Backpropagation()
nInputs = len(trainingSet[0]) - nOutputs
network = backpropagation.initialize(nInputs, nHiddenLayer, nOutputs)
backpropagation.trainingNetwork(network, trainingSet, learningRate, nEpochs, nOutputs, expectedError)

input('\nPress enter to view Result...')

print('\n################################ BACKPROPAGATION - RESULT #################################')
for row in testSet:
    prediction = backpropagation.predict(network, row)
    # print('Input =', (row), 'Expected = ', backpropagation.getExpected(row, nOutputs), 'Result =', (prediction))
    print('Expected = ', backpropagation.getExpected(row, nOutputs), 'Result =', (prediction))

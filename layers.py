import logging

import numpy as np

import functions as fc


class Layer:
    def __str__(self):
        return np.array_str(self.weights)

    def __mul__(self, other):
        output = map(self.activation[0], self.weights.dot(other)[0])
        return np.array(list(output))

    def process(self, other):
        output = np.array([other]).dot(self.weights)[0]
        output = np.add(output, self.bias)
        output = map(self.activation[0], output)
        return np.array(list(output))

    def learn(self, target):
        logging.debug('Output : {0}'.format(self.last_outputs))
        logging.debug('Target : {0}'.format(target.tolist()))
        logging.debug('Input  : {0}'.format(self.last_inputs))
        logging.debug('Weights: {0}'.format(self.weights.tolist()))

        eb, ew = 0, np.zeros((self.inputs, self.outputs))
        et = np.zeros((self.inputs, self.outputs))
        for y, line in enumerate(self.weights):
            for x in range(len(line)):
                input = self.last_inputs[y]
                output = self.last_outputs[x]
                t = sum(target[x])
                logging.debug('({0})({1})({2})'.format(t, output * (1 - output), input))    #last:   error = (expected - output) * transfer_derivative(output)
                ew[y, x] = t * output * (1 - output) * input								#hidden: error = (weight_k * error_j) * transfer_derivative(output)
                et[y, x] += t * output * (1 - output) * self.weights[y, x]
                if x == y:
                    logging.debug('{0}: ({1})({2})'.format(y, target[y][y], output * (1 - output)))
                    eb += target[y][y] * output * (1 - output)

        logging.debug('Eb: {0}'.format(eb))
        logging.debug('Ew: {0}'.format(ew.tolist()))
        logging.debug('Et: {0}'.format(et.tolist()))
        self._adjust_weights(ew, eb)
        return et

    def _adjust_weights(self, ew, eb):
        self.weights = np.subtract(self.weights, np.multiply(self.learning_rate, ew))  # learning rate * error * input
        self.bias = self.bias - self.learning_rate * eb	# ok


class Dense(Layer):
    def __init__(self, inputs, outputs, activation='log', bias=0.5, learning_rate=0.01):
        self.inputs = inputs
        self.outputs = outputs
        self.last_outputs = np.zeros(outputs)
        self.last_inputs = np.zeros(inputs)
        self.activation = fc.function_switcher(activation)
        self.weights = np.random.rand(inputs, outputs)
        self.bias = bias
        self.learning_rate = learning_rate


class Input(Layer):
    def __init__(self, inputs):
        pass

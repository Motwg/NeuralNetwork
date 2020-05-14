import logging

import numpy as np

from tools import functions, bias_init


class Layer:
    def __str__(self):
        return np.array_str(self.weights) + '\n' + np.array_str(self.bias)

    def __mul__(self, other):
        output = map(self.activation[0], self.weights.dot(other)[0])
        return np.array(list(output))

    def call(self, other):
        logging.debug('Input  : {0}'.format(other))
        output = np.array([other]).dot(self.weights)[0]
        output = np.add(output, self.bias)
        output = map(self.activation[0], output)
        output = list(output)
        logging.debug('Output : {0}'.format(output))
        return np.array(output)

    def learn(self, target):
        logging.debug('Input  : {}'.format(self.last_inputs))
        logging.debug('Output : {}'.format(self.last_outputs))
        logging.debug('Target : \n{}'.format(target))
        logging.debug('Weights: \n{}'.format(self.weights))

        errors_weights = np.zeros((self.inputs, self.units))
        errors_bias = np.zeros(self.units)
        for y, line in enumerate(self.weights):
            for x in range(len(line)):
                output = self.last_outputs[x]
                input = self.last_inputs[y]
                t = sum(target[x, :])
                derivative = self.activation[1](output)
                logging.debug('derivative: {:.4f}'.format(derivative))
                logging.debug('target: {:.4f}'.format(t))
                logging.debug('input: {:.4f}'.format(input))
                logging.debug('weight: {:.4f}'.format(self.weights[y, x]))
                errors_weights[y, x] += t * derivative * input
                errors_bias[x] += t * derivative * 1
                logging.debug('errors_weights: \n{}'.format(errors_weights))
                logging.debug('errors_bias   : {}'.format(errors_bias))
        self._adjust_weights(errors_weights, errors_bias)
        return errors_weights

    def _adjust_weights(self, errors_weights, errors_bias):
        # weights
        adjustment = np.multiply(self.learning_rate, errors_weights)
        logging.debug('Weights: {}'.format(self.weights.tolist()))
        logging.debug('Adjusts: {}'.format(adjustment.tolist()))
        self.weights = np.subtract(self.weights, np.array(adjustment))
        logging.debug('Weights: {}'.format(self.weights.tolist()))

        # bias
        adjustment = np.multiply(self.learning_rate, errors_bias)
        logging.debug('Bias: {}'.format(self.bias))
        logging.debug('Adjusts: {}'.format(adjustment.tolist()))
        self.bias = np.subtract(self.bias, np.array(adjustment))
        logging.debug('Bias: {}'.format(self.bias))


class Dense(Layer):
    def __init__(self, inputs, units, activation='log', bias_initializer='zeros', learning_rate=0.01):
        self.inputs = inputs
        self.units = units
        self.last_outputs = np.zeros(units)
        self.last_inputs = np.zeros(inputs)

        self.activation = functions.function_switcher(activation)
        self.weights = np.random.rand(inputs, units)
        self.bias = bias_init.bias_switcher(bias_initializer)(units)
        self.learning_rate = learning_rate


class Input(Layer):
    def __init__(self, inputs):
        pass

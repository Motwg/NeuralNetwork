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
        logging.debug('Input  : {0}'.format(other))
        output = np.array([other]).dot(self.weights)[0]
        output = np.add(output, self.bias)
        output = map(self.activation[0], output)
        output = np.array(list(output))
        logging.debug('Output : {0}'.format(output.tolist()))
        return output

    def learn(self, target):
        logging.debug('Output : {0}'.format(self.last_outputs))
        logging.debug('Target : {0}'.format(target.tolist()))
        logging.debug('Input  : {0}'.format(self.last_inputs))
        logging.debug('Weights: {0}'.format(self.weights.tolist()))
        eb, errors = 0, np.zeros((self.inputs, self.outputs))
        for y, line in enumerate(self.weights):
            for x in range(len(line)):
                output = self.last_outputs[x]
                input = self.last_inputs[y]
                t = sum(target[x, :])

                derivative = self.activation[1](output)
                logging.debug('derivative: {}'.format(derivative))
                logging.debug('target_x: {}'.format(t))
                logging.debug('weight: {}'.format(self.weights[y, x]))
                errors[y, x] += t * derivative * input
                eb += t * derivative
                logging.debug('errors: {}'.format(errors.tolist()))
        self._adjust_weights(errors, eb)
        return errors

    def _adjust_weights(self, errors, eb):
        # print('er: ', errors)
        adjustment = np.multiply(self.learning_rate, errors)
        # adjustment = np.multiply(adjustment, np.array([self.last_inputs]).T)
        # print(self.weights)
        # print('ad: ', np.array([adjustment]).T)
        logging.debug('Weights: {0}'.format(self.weights.tolist()))
        logging.debug('Adjusts: {0}'.format(adjustment.tolist()))
        self.weights = np.subtract(self.weights, np.array(adjustment))
        logging.debug('Weights: {0}'.format(self.weights.tolist()))
        self.bias -= self.learning_rate * eb


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

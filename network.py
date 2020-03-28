import logging

import numpy as np

import layers as lr

'''
logging.basicConfig(level=logging.DEBUG, filename='files/network.log', filemode='w',
                    format='%(levelname)-8s %(funcName)-20s %(message)s')
'''


def normalise(vector):
    print(vector)
    return [(v - min(vector)) / (max(vector) - min(vector)) for v in vector]


class Network:
    def __init__(self, layers, use_normalise=True):
        assert isinstance(layers, list)
        float_formatter = "{:.4f}".format
        np.set_printoptions(formatter={'float_kind': float_formatter})
        self.layers = layers
        self.error = 0
        self.use_normalise = use_normalise

    def __str__(self):
        return ''.join([str(layer) + '\n' for layer in self.layers])

    def run_once(self, test_vector, test_expected):
        if self.use_normalise:
            output = np.array(normalise(test_vector.copy()))
        else:
            output = np.array(test_vector.copy())
        for layer in self.layers:
            layer.last_inputs = np.array(output.copy())
            if self.use_normalise:
                output = np.array(normalise(layer.process(output)))
            else:
                output = layer.process(output)
            layer.last_outputs = output
        self.error = 0.5 * sum([(o - t) * (o - t) for o, t in zip(output, test_expected)])
        return output

    def learn(self, test_vector, test_expected):
        output = self.run_once(test_vector, test_expected)
        self._change_weights(test_expected)
        return output

    def _change_weights(self, test_expected):
        target = np.subtract(self.layers[-1].last_outputs, test_expected.copy())
        target = np.diag(target)
        for layer in reversed(self.layers):
            logging.debug('Layer: {}'.format(self.layers.index(layer)))
            target = layer.learn(target)


if __name__ == '__main__':
    x = [1, 4, 5]
    first = lr.Dense(3, 2, activation='relu', learning_rate=.005)
    second = lr.Dense(2, 2, activation='relu', learning_rate=.005)
    net = Network([
        first,
        second
    ])

    first.weights = np.array([[0.1, 0.2],
                              [0.3, 0.4],
                              [0.5, 0.6]])
    second.weights = np.array([[0.7, 0.8],
                               [0.9, 0.1]])

    print('Input:  ', x)
    print('Weights:\n', net)
    for i in range(1500):
        print('Output: ', net.learn(x, [1, 0]))
        print('Error:  ', net.error)
    print('Weights:\n', net)

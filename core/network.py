import logging

import numpy as np

from core import layers as lr, visualisation as vis


def normalise(vector):
    return [(v - min(vector)) / (max(vector) - min(vector)) for v in vector]


class Network:
    def __init__(self, layers, use_normalise=False):
        assert isinstance(layers, list)
        float_formatter = "{:.4f}".format
        np.set_printoptions(formatter={'float_kind': float_formatter})
        self.layers = layers
        self.error = 0
        self.use_normalise = use_normalise
        self.iteration = 0
        self.learn_after = 1

    def __str__(self):
        return ''.join([str(layer) + '\n' for layer in self.layers])

    def run_once(self, test_vector, test_expected):
        if self.use_normalise:
            output = np.array(normalise(test_vector.copy()))
        else:
            output = np.array(test_vector.copy())
        for layer in self.layers:
            layer.last_inputs = np.array(output.copy())
            output = layer.call(output)
            layer.last_outputs = output
        self.error = 0.5 * sum([(o - t) * (o - t) for o, t in zip(output, test_expected)])
        return output

    def learn(self, test_vector, test_expected):
        output = self.run_once(test_vector, test_expected)
        target = np.subtract(self.layers[-1].last_outputs, test_expected)
        target = np.diag(target)
        for layer in reversed(self.layers):
            logging.debug('Layer: {}'.format(self.layers.index(layer)))
            target = layer.learn(target)

        self.iteration += 1
        if self.iteration >= self.learn_after:
            self._change_weights()
        return output

    def _change_weights(self):
        for layer in self.layers:
            layer.adjust_weights()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename='../logs/network.log', filemode='w',
                        format='%(levelname)-8s %(funcName)-20s %(message)s')

    x = [1, 4, 5]
    first = lr.Dense(3, 2, activation='none', learning_rate=.1)
    second = lr.Dense(2, 2, activation='log', learning_rate=.2)
    net = Network([
        first,
        second
    ])

    '''
    first.weights = np.array([[0.1, 0.2],
                              [0.3, 0.4],
                              [0.5, 0.6]])
    second.weights = np.array([[0.7, 0.8],
                               [0.9, 0.1]])
    '''

    print('Input:  ', x)
    print('Weights:\n', net)
    err = []
    for i in range(200):
        print('Output: ', net.learn(x, [1, 0]))
        print('Error:  ', net.error)
        err.append(net.error)
    print('Weights:\n', net)
    vis.show(err)


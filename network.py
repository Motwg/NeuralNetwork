import numpy as np

import layers as lr

'''
logging.basicConfig(level=logging.DEBUG, filename='files/network.log', filemode='w',
                    format='%(levelname)-8s %(funcName)-20s %(message)s')
'''


def normalise(vector):
    return [(v - min(vector)) / (max(vector) - min(vector)) for v in vector]


class Network:
    def __init__(self, layers):
        assert isinstance(layers, list)
        self.layers = layers
        self.error = 0

    def __str__(self):
        return ''.join([str(layer) + '\n' for layer in self.layers])

    def run_once(self, test_vector, test_expected):
        output = np.array(normalise(test_vector.copy()))
        # output = np.array(test_vector.copy())
        for layer in self.layers:
            layer.last_inputs = np.array(output.copy())
            output = layer.process(output)
            layer.last_outputs = output
        self.error = 0.5 * sum([(o - t) * (o - t) for o, t in zip(output, test_expected)])
        return output

    def _change_weights(self, test_expected):
        target = np.subtract(self.layers[-1].last_outputs, test_expected.copy())
        target = np.diag(target)
        for layer in reversed(self.layers):
            target = layer.learn(target)

    def learn(self, test_vector, test_expected):
        output = self.run_once(test_vector, test_expected)
        self._change_weights(test_expected)
        return output


if __name__ == '__main__':
    x = [1, 4, 5]
    first = lr.Dense(3, 2, activation='log')
    second = lr.Dense(2, 2, activation='log')
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
    print('Output: ', net.learn(x, [0.1, 0.05]))
    print('Weights:\n', net)
    print('Error:  ', net.error)
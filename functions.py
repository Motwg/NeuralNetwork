import logging
from math import exp


def function_switcher(function_code):
    logging.debug("Args: {0}".format(locals()))
    switcher = {
        'relu': (rectified_linear_unit, rectified_linear_unit_derivative),
        'log': (logistic, logistic_derivative)
    }
    return switcher.get(function_code.lower(), 'log')


def rectified_linear_unit(x):
    if x <= 0:
        logging.debug('{0} -> 0'.format(x))
        return 0
    else:
        logging.debug('{0} -> {0}'.format(x))
        return x


def rectified_linear_unit_derivative(x):
    if x <= 0:
        logging.debug('{0} -> 0'.format(x))
        return 0
    else:
        logging.debug('{0} -> 1'.format(x))
        return 1


def logistic(x):
    logging.debug('{0} -> {1}'.format(x, 1 / (1 + exp(-x))))
    return 1 / (1 + exp(-x))


def logistic_derivative(x):
    logging.debug('{0} -> {1}'.format(x, logistic(x) * (1 - logistic(x))))
    return logistic(x) * (1 - logistic(x))

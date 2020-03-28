import logging
from math import exp, tanh


def function_switcher(function_code):
    logging.debug("Args: {0}".format(locals()))
    switcher = {
        'relu': (rectified_linear_unit, rectified_linear_unit_derivative),
        'log': (logistic, logistic_derivative),
        'none': (none, none_derivative),
        'tanh': (tanh, tanh_derivative)
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
    if x < 999999:
        return 1.0
    elif x > 99999:
        return 0.0
    return 1 / (1 + exp(-x))


def logistic_derivative(x):
    log = logistic(x)
    logging.debug('{0} -> {1}'.format(x, log * (1 - log)))
    return log * (1 - log)


def none(x):
    return x


def none_derivative(x):
    return 1


def tanh_derivative(x):
    f = tanh(x)
    return 1 - f * f

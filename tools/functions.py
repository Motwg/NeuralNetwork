# from math import exp, tanh
from numpy import exp, tanh


def function_switcher(function_code):
    switcher = {
        'relu': (rectified_linear_unit, rectified_linear_unit_derivative),
        'log': (logistic, logistic_derivative),
        'sig': (logistic, logistic_derivative),
        'none': (none, none_derivative),
        'tanh': (tanh, tanh_derivative)
    }
    return switcher.get(function_code.lower(), 'log')


def rectified_linear_unit(x):
    if x <= 0:
        return 0
    else:
        return x


def rectified_linear_unit_derivative(x):
    if x <= 0:
        return 0
    else:
        return 1


def logistic(x):
    return 1 / (1 + exp(-x))


def logistic_derivative(x):
    log = logistic(x)
    return log * (1 - log)


def none(x):
    return x


def none_derivative(x):
    return 1


def tanh_derivative(x):
    f = tanh(x)
    return 1 - f * f

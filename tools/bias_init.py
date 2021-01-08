import numpy as np


def bias_switcher(bias_code):
    switcher = {
        'zeros': zeros,
        'random': random
    }
    return switcher.get(bias_code.lower(), 'zeros')


def random(outputs):
    return np.random.rand(outputs)


def zeros(outputs):
    return np.zeros(outputs)

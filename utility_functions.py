import math
import numpy as np
import tensorflow as tf


def power_utility(x,b):
    return tf.pow(x, b) / b


def hara_utility(x, b=0.5, F=10):
    if isinstance(x, np.ndarray):
        V_t = x[0]
    else:
        V_t = x

    if V_t >= F:
        return math.pow(1 / (1 - b) * (V_t - F), b) * (1 - b) / b

    m = math.pow(1 / (1 - b), b) * (1 - b) / b
    return m * (V_t - F)

from numpy import random as rnd
from math import sin
import math
def Levy(d):
    beta = 3 / 2

    # Eq. (3.10) check the equation

    sigma = ((rnd.gamma(1 + beta) * sin(math.pi * beta / 2) / 
        (rnd.gamma((1 + beta) / 2) * beta *
        2 ** ((beta - 1) / 2))) ** (1 / beta))

    # check it out
    u = rnd.normal(1, d) * sigma
    v = rnd.normal(1, d)
    
    # check it out
    step = u / abs(v) ** (1 / beta)

    # Eq. (3.9)
    o = 0.01 * step

    return o

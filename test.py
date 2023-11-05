from generate import Distribution
from enum import Enum


val = Distribution.POISSONIAN

print(val == Distribution.POISSONIAN)


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

color = Color.RED



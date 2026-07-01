import numpy as np

from config import *


def random_position():

    return np.random.uniform(
        -BOUNDARY,
        BOUNDARY,
        size=2
    )


def random_scenario():

    while True:

        start = random_position()

        target = random_position()

        dist = np.linalg.norm(target - start)

        if MIN_START_DISTANCE < dist < MAX_START_DISTANCE:

            return start, target


def distance(a, b):

    return np.linalg.norm(a - b)


def clamp_position(pos):

    return np.clip(
        pos,
        -BOUNDARY,
        BOUNDARY
    )
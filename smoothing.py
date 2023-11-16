from numba import cuda
from math import pi


SMOOTHING_RADIUS = 0.35


def cpu_smoothing_function(dist):
    if dist >= SMOOTHING_RADIUS:
        return 0
    
    volume = (pi * SMOOTHING_RADIUS**4) / 6
    return (SMOOTHING_RADIUS - dist) * (SMOOTHING_RADIUS - dist) / volume


def cpu_smoothing_derivative(dist):
    if dist >= SMOOTHING_RADIUS:
        return 0
    
    scale = 12 / (SMOOTHING_RADIUS**4 * pi)
    return (dist - SMOOTHING_RADIUS) * scale


@cuda.jit
def gpu_smoothing_function(dist):
    if dist >= SMOOTHING_RADIUS:
        return 0
    
    volume = (pi * SMOOTHING_RADIUS**4) / 6
    return (SMOOTHING_RADIUS - dist) * (SMOOTHING_RADIUS - dist) / volume


@cuda.jit
def gpu_smoothing_derivative(dist):
    if dist >= SMOOTHING_RADIUS:
        return 0
    
    scale = 12 / (SMOOTHING_RADIUS**4 * pi)
    return (dist - SMOOTHING_RADIUS) * scale
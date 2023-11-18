from numba import cuda
from math import pi


SMOOTHING_RADIUS = 0.35


@cuda.jit
def smoothing_function(dist):
    if dist >= SMOOTHING_RADIUS:
        return 0
    
    volume = (pi * SMOOTHING_RADIUS**4) / 6
    return (SMOOTHING_RADIUS - dist) * (SMOOTHING_RADIUS - dist) / volume


@cuda.jit
def smoothing_derivative(dist):
    if dist >= SMOOTHING_RADIUS:
        return 0
    
    scale = 12 / (SMOOTHING_RADIUS**4 * pi)
    return (dist - SMOOTHING_RADIUS) * scale


@cuda.jit
def visc_smoothing(dist):
    if dist >= SMOOTHING_RADIUS:
        return 0
    
    volume = pi * SMOOTHING_RADIUS**8 / 4
    value = SMOOTHING_RADIUS * SMOOTHING_RADIUS - dist * dist

    return value**3 / volume


@cuda.jit
def near_density_smoothing(dist):
    if dist >= SMOOTHING_RADIUS:
        return 0
    
    volume = pi * SMOOTHING_RADIUS**4 / 2
    diff = SMOOTHING_RADIUS - dist

    return diff * diff * diff / volume


@cuda.jit
def near_density_derivative(dist):
    if dist >= SMOOTHING_RADIUS:
        return 0
    
    volume = pi * SMOOTHING_RADIUS**4 / 2
    coefficient = 3 / volume
    diff = SMOOTHING_RADIUS - dist

    return coefficient * diff * diff
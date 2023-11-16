"""
Not used
"""


import smoothing
from numba import cuda
from math import sqrt


@cuda.jit
def draw_circle(pos, colour, new_colour, screen_array):
    radius = 5#smoothing.SMOOTHING_RADIUS
    x, y = pos
    for i in range(-radius, radius):
        for j in range(-radius, radius):
            inx1 = i + x
            inx2 = j + y

            if not 0 <= inx1 < len(screen_array[0]) or not 0 <= inx2 < len(screen_array[1]):
                continue

            dist = i**2 + j**2

            multiplier = 1#smoothing.gpu_smoothing_function(sqrt(dist))# / radius**3

            if dist > radius**2 or multiplier < 0:
                continue

            mul_array(colour, multiplier, new_colour)

            screen_array[inx1][inx2] += convert_colour(new_colour)

            if screen_array[inx1][inx2] > 16777215:
                screen_array[inx1][inx2] = 16777215


@cuda.jit
def mul_array(array, scalar, result):
    for i, x in enumerate(array):
        result[i] = x * scalar


@cuda.jit
def update_screen_array(positions, colours, screen_array, blank_colour):
    inx = cuda.grid(1)

    if inx >= len(positions):
        return

    pos = positions[inx]
    colour = colours[inx]

    draw_circle(pos, colour, blank_colour, screen_array)


@cuda.jit
def convert_colour(colour):
    final_colour = 0
    for i, x in enumerate(colour):
        final_colour += x << (i * 8)

    return final_colour
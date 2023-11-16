import pygame
import numpy
from numba import cuda
from Graphics import menu
from Physics import constants, precalculate


NUM_THREADS = 4


pygame.init()
pygame.display.set_caption("Fluid Dynamics")


window = pygame.display.set_mode((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))
clock = pygame.time.Clock()


def draw_screen(screen_array):
    pygame.surfarray.blit_array(window, screen_array)

    menu.update()

    pygame.display.update()


def prepare_gpu_data(positions, colours, screen_array):
    num_blocks = (NUM_THREADS + len(positions) - 1) // NUM_THREADS

    blank_colour = numpy.zeros(3, numpy.int64)

    pos_d = cuda.to_device(positions)
    colour_d = cuda.to_device(colours)
    blank_c_d = cuda.to_device(blank_colour)
    screen_d = cuda.to_device(screen_array)

    return num_blocks, pos_d, colour_d, blank_c_d, screen_d


def convert_pos(pos):
    x, y = pos

    x -= constants.BOUNDING_LEFT
    y -= constants.BOUNDING_TOP

    scale_x = constants.SCREEN_WIDTH / (constants.BOUNDING_RIGHT - constants.BOUNDING_LEFT)
    scale_y = constants.SCREEN_HEIGHT / (constants.BOUNDING_BOTTOM - constants.BOUNDING_TOP)

    return [x * scale_x, y * scale_y]


def draw_circles(positions, colours):
    window.fill((0, 0, 0))

    for i, x in zip(positions, colours):
        pygame.draw.circle(window, x, i, 5)

    menu.update()

    pygame.display.update()


def draw_particles(positions, colours):
    if not menu.init:
        menu.create_sliders(window)

    new_pos = [convert_pos(i) for i in positions]
    new_pos = numpy.array(new_pos, numpy.int64)
    
    #TODO: maybe use blit_array() for speedups
    draw_circles(new_pos, colours)

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            quit()
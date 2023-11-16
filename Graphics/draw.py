import pygame
import numpy
import constants
from Graphics import menu


pygame.init()
pygame.display.set_caption("Fluid Dynamics")


window = pygame.display.set_mode((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))
clock = pygame.time.Clock()


def draw_screen(screen_array):
    pygame.surfarray.blit_array(window, screen_array)

    menu.update()

    pygame.display.update()


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
        menu.create_elements(window)

    new_pos = [convert_pos(i) for i in positions]
    new_pos = numpy.array(new_pos, numpy.int64)
    
    #TODO: maybe use blit_array() for speedups
    draw_circles(new_pos, colours)

    restart = menu.restart.button_object.clicked

    return restart
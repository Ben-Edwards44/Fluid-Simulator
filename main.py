import Graphics.draw
import Graphics.colours
import Physics.particle
import Physics.cpu
import numpy
from math import sqrt
from random import uniform
import pygame

from time import time


SPACE_X = 0.25
SPACE_Y = 0.25


FPS = 120


def create_particles(num):
    particles = []

    #"""    
    square = int(sqrt(num))
    remainder = num - square**2

    offset_x = square * SPACE_X / 2
    offset_y = square * SPACE_Y / 2

    for x in range(square):
        for y in range(square):
            particle = Physics.particle.Particle((x * SPACE_X - offset_x, y * SPACE_Y - offset_y), (255, 255, 255), x * 8 + y)
            particles.append(particle)

    for i in range(remainder):
        particle = Physics.particle.Particle((i * SPACE_X - offset_x, square * SPACE_Y - offset_y), (255, 255, 255), num - remainder + i)
        particles.append(particle)
    """

    for i in range(num):
        x = uniform(-10, 10)
        y = uniform(-10, 10)

        particles.append(Physics.particle.Particle((x, y), (255, 255, 255), i))
    """

    return particles


def create_arrays(particles):
    pos = []
    vel = []
    cols = []

    for i in particles:
        pos.append([i.x, i.y])
        vel.append([i.vel_x, i.vel_y])
        cols.append(i.colour)

    return numpy.array(pos, float), numpy.array(vel, float), numpy.array(cols)


def get_mouse_stats():
    clicked = pygame.mouse.get_pressed()[0]
    x, y = pygame.mouse.get_pos()

    return clicked, x, y


def main():
    particles = create_particles(1024)
    positions, vels, colours = create_arrays(particles)

    Physics.cpu.setup(vels)

    clock = pygame.time.Clock()
    while True:
        start = time()
        delta_time = clock.tick(FPS) / 1000

        if delta_time > 0.5:
            delta_time = 1 / FPS

        mouse_clicked, mouse_x, mouse_y = get_mouse_stats()
        positions, vels = Physics.cpu.update_all_particles(positions, delta_time, mouse_clicked, mouse_x, mouse_y)
        colours = Graphics.colours.get_colours(vels, colours)

        Graphics.draw.draw_particles(positions, colours)


if __name__ == "__main__":
    main()
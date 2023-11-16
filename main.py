import constants
import Graphics.draw
import Graphics.colours
from Physics import cpu, particle
import numpy
from math import sqrt
import pygame


def create_particles(num):
    particles = []

    square = int(sqrt(num))
    remainder = num - square**2

    offset_x = square * constants.SPACE_X / 2
    offset_y = square * constants.SPACE_Y / 2

    for x in range(square):
        for y in range(square):
            p = particle.Particle((x * constants.SPACE_X - offset_x, y * constants.SPACE_Y - offset_y), (255, 255, 255), x * 8 + y)
            particles.append(p)

    for i in range(remainder):
        p = particle.Particle((i * constants.SPACE_X - offset_x, square * constants.SPACE_Y - offset_y), (255, 255, 255), num - remainder + i)
        particles.append(p)

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


def setup():
    global positions, vels, colours

    particles = create_particles(constants.NUM_PARTICLES)
    positions, vels, colours = create_arrays(particles)

    cpu.setup(vels)


def reset():
    global positions, vels

    particles = create_particles(constants.NUM_PARTICLES)
    new_positions = [[i.x, i.y] for i in particles]

    for i in range(len(positions)):
        for x in range(2):
            positions[i][x] = new_positions[i][x]
            vels[i][x] = 0

    cpu.setup(vels)


def main():
    global positions, vels, colours
    
    setup()

    clock = pygame.time.Clock()
    while True:
        delta_time = clock.tick(constants.FPS) / 1000

        if delta_time > 0.5:
            delta_time = 1 / constants.FPS

        mouse_clicked, mouse_x, mouse_y = get_mouse_stats()
        positions, vels = cpu.update_all_particles(positions, delta_time, mouse_clicked, mouse_x, mouse_y)
        colours = Graphics.colours.get_colours(vels, colours)

        restart = Graphics.draw.draw_particles(positions, colours)

        if restart:
            reset()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                quit()



if __name__ == "__main__":
    main()
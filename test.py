import Physics.subdivide as subdivide
import Physics.my_subdivide as my
from Physics import constants
import pygame
from random import uniform
from numpy import array, zeros, int64
from numba import cuda
import smoothing


NUM = 1000


pygame.init()
window = pygame.display.set_mode((500, 500))


def create_particles():
    particles = []

    for _ in range(NUM):
        x = uniform(constants.BOUNDING_LEFT, constants.BOUNDING_RIGHT)
        y = uniform(constants.BOUNDING_TOP, constants.BOUNDING_BOTTOM)
    
        particles.append([x, y])

    #l = NUM**0.5
    #for i in range(NUM):
    #    x = i // l * 0.2 - 5
    #    y = i % l * 0.2 - 5
    #    particles.append([x, y])

    return array(particles)


def is_in(arr, pos):
    for i in arr:
        if i[0] == pos[0] and i[1] == pos[1]:
            return True
        
    return False


def draw(all, nearby, inx):
    #expect nearby and all as lists of positions
    window.fill((0, 0, 0))

    for i in all:
        pygame.draw.circle(window, (255, 0, 0), convert_pos(i), 10)

    for i in nearby:
        pygame.draw.circle(window, (0, 255, 0), convert_pos(i), 10)

    scale = 500 / (constants.BOUNDING_RIGHT - constants.BOUNDING_LEFT)

    for i in range(0, 500, int(smoothing.SMOOTHING_RADIUS * scale)):
        pygame.draw.line(window, (255, 255, 255), (i, 0), (i, 500), 2)
    for i in range(0, 500, int(smoothing.SMOOTHING_RADIUS * scale)):
        pygame.draw.line(window, (255, 255, 255), (0, i), (500, i), 2)

    pygame.draw.circle(window, (255, 255, 255), convert_pos(all[inx]), smoothing.SMOOTHING_RADIUS * scale, 2)

    pygame.display.update()


def get_particle_inx(pos):
    x, y = unconvert_pos(pygame.mouse.get_pos())

    min_dist = None
    min_inx = None

    for j, i in enumerate(pos):
        dist = (i[0] - x)**2 + (i[1] - y)**2

        if min_dist == None or dist < min_dist:
            min_dist = dist
            min_inx = j

    return min_inx


def all_near(inx, near_all, length, positions):
    pos = []

    for i in near_all[inx]:
        if i != length:
            pos.append(positions[i])
        else:
            break

    return pos


def get_nearby(inx, pos):
    p_d = cuda.to_device(pos)

    near_inxs = array([len(pos) for _ in pos])
    n_d = cuda.to_device(near_inxs)

    s_look, s_inx = subdivide.get_device_data(pos, p_d, 32, 64)

    subdivide.get_nearby_particles[1, 1](inx, p_d, s_look, s_inx, n_d)

    near_inxs = n_d.copy_to_host()

    positions = []
    for i in near_inxs:
        if i != len(pos):
            positions.append(pos[i])
        else:
            break

    return positions


def precalculate_positions(length, positions, positions_d, num_blocks):
    temp_array = array([length for _ in positions])
    result_inxs = array([[length for _ in positions] for _ in positions])
    r = cuda.to_device(result_inxs)

    spatial_lookup, start_indices = subdivide.get_device_data(positions, positions_d, 32, num_blocks)
    subdivide.cpu_get_all_particle_pos(length, positions, spatial_lookup.copy_to_host(), start_indices.copy_to_host(), temp_array, result_inxs)

    a = r.copy_to_host()

    return result_inxs


def convert_pos(pos):
    x, y = pos

    x -= constants.BOUNDING_LEFT
    y -= constants.BOUNDING_TOP

    scale_x = 500 / (constants.BOUNDING_RIGHT - constants.BOUNDING_LEFT)
    scale_y = 500 / (constants.BOUNDING_BOTTOM - constants.BOUNDING_TOP)

    return [x * scale_x, y * scale_y]


def unconvert_pos(pos):
    x, y = pos

    scale_x = 500 / (constants.BOUNDING_RIGHT - constants.BOUNDING_LEFT)
    scale_y = 500 / (constants.BOUNDING_BOTTOM - constants.BOUNDING_TOP)

    x /= scale_x
    y /= scale_y

    return x + constants.BOUNDING_LEFT, y + constants.BOUNDING_TOP


def new_pos(pos, pos_d):
    result_d = my.precompute_nearby_inxs(pos, pos_d, 128)
    #r = result_d.copy_to_host()

    return result_d


def new_get_near(inx, positions, precomputed):
    result = array([len(positions) for _ in positions])
    result_d = cuda.to_device(result)

    pos_d = cuda.to_device(positions)

    my.find_near_from_precompute[1, 1](inx, pos_d, precomputed, result_d)

    near_inxs = result_d.copy_to_host()

    near_pos = []
    for i in near_inxs:
        if i == len(positions):
            break
        else:
            near_pos.append(positions[i])

    return near_pos


def main():
    pos = create_particles()
    p_d = cuda.to_device(pos)
    near_all = new_pos(pos, p_d)

    while True:
        inx = get_particle_inx(pos)
        near = new_get_near(inx, pos, near_all)
        print(my.cpu_find_cell(pos[inx][0], pos[inx][1]))

        draw(pos, near, inx)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                quit()


main()
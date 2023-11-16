"""
Unused at the moment because it wont work
"""


from Physics import constants
from smoothing import SMOOTHING_RADIUS
from numba import cuda
from numpy import array


CELLS_X = int((constants.BOUNDING_RIGHT - constants.BOUNDING_LEFT) // SMOOTHING_RADIUS)
CELLS_Y = int((constants.BOUNDING_BOTTOM - constants.BOUNDING_TOP) // SMOOTHING_RADIUS)


def cpu_find_cell(x, y):
    offset_x = x - float(constants.BOUNDING_LEFT)
    offset_y = y - float(constants.BOUNDING_TOP)

    cell_x = int(offset_x / SMOOTHING_RADIUS)
    cell_y = int(offset_y / SMOOTHING_RADIUS)

    return cell_x, cell_y


@cuda.jit
def find_cell(x, y):
    offset_x = x - float(constants.BOUNDING_LEFT)
    offset_y = y - float(constants.BOUNDING_TOP)

    cell_x = int(offset_x / SMOOTHING_RADIUS)
    cell_y = int(offset_y / SMOOTHING_RADIUS)

    return cell_x, cell_y


def get_nearby_inxs(inx, positions, result_array):
    length = len(positions)

    if inx >= length:
        return
    
    x, y = positions[inx]
    cell_x, cell_y = cpu_find_cell(x, y)

    if cell_x < 0:
        cell_x = 0
    elif cell_x > CELLS_X:
        cell_x = CELLS_X - 1

    if cell_y < 0:
        cell_y = 0
    elif cell_y > CELLS_Y:
        cell_y = CELLS_Y - 1

    place_inx = 0
    while result_array[cell_x][cell_y][place_inx] != length and place_inx < length:
        place_inx += 1

    return cell_x, cell_y, place_inx


def precompute_nearby_inxs(positions, positions_d, num_blocks):
    #run on cpu
    length = len(positions)

    cell_data = array([[[length for _ in range(length)] for _ in range(CELLS_Y)] for _ in range(CELLS_X)])

    for inx in range(length):
        x, y, place = get_nearby_inxs(inx, positions, cell_data)
        cell_data[x][y][place] = inx

    cell_data_d = cuda.to_device(cell_data)

    particle_data = array([[length for _ in range(length)] for _ in range(length)])
    particle_d = cuda.to_device(particle_data)

    find_near_from_precompute[num_blocks, constants.NUM_THREADS](positions_d, cell_data_d, particle_d)

    return particle_d


@cuda.jit
def find_near_from_precompute(positions, all_nearby_pos, result_array):
    #store nearby particle indexes in result_array

    inx = cuda.grid(1)
    length = len(positions)

    if inx >= length:
        return

    rad_sq = SMOOTHING_RADIUS * SMOOTHING_RADIUS

    x, y = positions[inx]
    cell_x, cell_y = find_cell(x, y)

    place_inx = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if not (0 <= cell_x + i < len(all_nearby_pos) and 0 <= cell_y + j < len(all_nearby_pos[0])):
                continue

            for particle_inx in all_nearby_pos[cell_x + i][cell_y + j]:
                if particle_inx == length:
                    break

                new_x, new_y = positions[particle_inx]
                dist_sq = (x - new_x)**2 + (y - new_y)**2

                if dist_sq <= rad_sq:
                    result_array[inx][place_inx] = particle_inx
                    place_inx += 1
"""
Unused at the moment because it wont work
"""


import constants
from smoothing import SMOOTHING_RADIUS
import numpy
from numba import cuda


PRIME_X = 977
PRIME_Y = 509

CELLS_X = int((constants.BOUNDING_RIGHT - constants.BOUNDING_LEFT) // SMOOTHING_RADIUS)
CELLS_Y = int((constants.BOUNDING_BOTTOM - constants.BOUNDING_TOP) // SMOOTHING_RADIUS)


@cuda.jit
def get_cell_keys(positions, cell_keys):
    inx = cuda.grid(1)
    length = len(positions)

    if inx >= length:
        return
    
    x, y = positions[inx]

    cell_x, cell_y = find_cell(x, y)
    cell_key = hash_cell(cell_x, cell_y, length)

    cell_keys[inx] = cell_key
    

@cuda.jit
def find_cell(x, y):
    offset_x = x - float(constants.BOUNDING_LEFT)
    offset_y = y - float(constants.BOUNDING_TOP)

    cell_x = int(offset_x / SMOOTHING_RADIUS)
    cell_y = int(offset_y / SMOOTHING_RADIUS)

    if cell_x >= CELLS_X:
        cell_x = CELLS_X - 1
    elif cell_x < 0:
        cell_x = 0

    if cell_y >= CELLS_Y:
        cell_y = CELLS_Y - 1
    elif cell_y < 0:
        cell_y = 0

    return cell_x, cell_y


@cuda.jit
def hash_cell(cell_x, cell_y, length):
    p_x = cell_x * PRIME_X
    p_y = cell_y * PRIME_Y

    hash = p_x + p_y

    return hash % length


def get_spatial_lookup(length, pos_d, num_threads, num_blocks):
    #run on cpu
    cell_keys = numpy.zeros(length, numpy.int64)
    cell_keys_d = cuda.to_device(cell_keys)

    get_cell_keys[num_blocks, num_threads](pos_d, cell_keys_d)

    cell_keys = cell_keys_d.copy_to_host()
    
    spatial_lookup = [[i, x] for i, x in enumerate(cell_keys)]
    spatial_lookup = sorted(spatial_lookup, key=lambda x : x[1])

    return numpy.array(spatial_lookup)


@cuda.jit
def start_index(spatial_lookup, start_indices):
    inx = cuda.grid(1)

    if inx >= len(spatial_lookup):
        return
    
    key = spatial_lookup[inx][1]

    if inx == 0:
        prev_key = len(spatial_lookup) + 1
    else:
        prev_key = spatial_lookup[inx - 1][1]
    
    if key != prev_key:
        start_indices[key] = inx


def get_device_data(length, pos_d, num_blocks):
    #run on cpu
    spatial_lookup = get_spatial_lookup(length, pos_d, constants.NUM_THREADS, num_blocks)
    spatial_lookup_d = cuda.to_device(spatial_lookup)

    start_indices = numpy.zeros(length, numpy.int64)
    start_indices_d = cuda.to_device(start_indices)

    start_index[num_blocks, constants.NUM_THREADS](spatial_lookup_d, start_indices_d)

    return spatial_lookup_d, start_indices_d


@cuda.jit
def get_nearby_particles(inx, positions, spatial_lookup, start_indices, result_array):
    #store nearby particle indices in result array
    rad_square = SMOOTHING_RADIUS * SMOOTHING_RADIUS

    x, y = positions[inx]
    cell_x, cell_y = find_cell(x, y)

    result_inx = 0

    for i in range(-1, 2):
        for j in range(-1, 2):
            current_x = cell_x + i
            current_y = j + cell_y

            if not 0 <= current_x <= CELLS_X or not 0 <= current_y <= CELLS_Y:
                continue

            key = hash_cell(current_x, current_y, len(positions))
            start_inx = start_indices[key]

            while start_inx < len(spatial_lookup) and spatial_lookup[start_inx][1] == key:
                p_inx = spatial_lookup[start_inx][0]

                dist = (positions[p_inx][0] - x)**2 + (positions[p_inx][1] - y)**2

                if dist <= rad_square:
                    result_array[result_inx] = p_inx
                    result_inx += 1

                start_inx += 1
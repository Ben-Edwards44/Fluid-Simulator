import constants
from Physics import gpu, precalculate, subdivide
from numba import cuda
import numpy


def setup(vel_array):
    global vels

    vels = vel_array


def precalculate_densities(length, vels_d, positions_d, num_blocks):
    densities = numpy.zeros(length)
    near_densities = numpy.zeros(length)
    density_d = cuda.to_device(densities)
    near_density_d = cuda.to_device(near_densities)

    predicted_pos = numpy.zeros((length, 2))
    predicted_d = cuda.to_device(predicted_pos)

    precalculate.predict_positions[num_blocks, constants.NUM_THREADS](positions_d, vels_d, predicted_d)

    spatial_lookup, start_indices = subdivide.get_device_data(length, predicted_d, num_blocks)
    result_arrays = numpy.full((length, length), length)
    result_d = cuda.to_device(result_arrays)

    precalculate.find_density_near_density[num_blocks, constants.NUM_THREADS](predicted_d, density_d, near_density_d, spatial_lookup, start_indices, result_d)

    return density_d, near_density_d, result_d


def prepare_gpu_data(positions):
    num_blocks = (constants.NUM_THREADS + len(positions) - 1) // constants.NUM_THREADS

    pos_d = cuda.to_device(positions)
    vels_d = cuda.to_device(vels)

    return num_blocks, pos_d, vels_d


def print_avg_density(densities):
    d = densities.copy_to_host()

    print(sum(d) / len(d))


def convert_mouse_pos(x, y):
    scale_x = constants.SCREEN_WIDTH / (constants.BOUNDING_RIGHT - constants.BOUNDING_LEFT)
    scale_y = constants.SCREEN_HEIGHT / (constants.BOUNDING_BOTTOM - constants.BOUNDING_TOP)

    x /= scale_x
    y /= scale_y

    x += constants.BOUNDING_LEFT
    y += constants.BOUNDING_TOP

    return x, y


def update_all_particles(positions, delta_time, mouse_clicked, mouse_x, mouse_y):
    global vels

    blocks, pos_d, vels_d = prepare_gpu_data(positions)

    density_d, near_density_d, nearby_d = precalculate_densities(len(positions), vels_d, pos_d, blocks)

    mouse_x, mouse_y = convert_mouse_pos(mouse_x, mouse_y)
    gpu.update_particles[blocks, constants.NUM_THREADS](pos_d, vels_d, density_d, near_density_d, nearby_d, constants.target_density, constants.pressure_multiplier, constants.near_pressure_multiplier, constants.visc_strength, mouse_clicked, mouse_x, mouse_y)

    vels = vels_d.copy_to_host()

    gpu.update_pos_from_vel[blocks, constants.NUM_THREADS](pos_d, vels_d, delta_time)

    positions = pos_d.copy_to_host()

    return positions, vels
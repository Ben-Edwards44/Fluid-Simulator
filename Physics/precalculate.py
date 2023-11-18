import smoothing
import constants
from Physics import subdivide
from numba import cuda
from math import sqrt


@cuda.jit
def predict_positions(positions, vels, result_array):
    inx = cuda.grid(1)

    if inx >= len(positions):
        return
    
    pos = positions[inx]
    vel = vels[inx]

    if constants.APPLY_GRAVITY:
        vel[1] += constants.GRAVITY * constants.PREDICT_D_T
    
    result_array[inx][0] = pos[0] + vel[0] * constants.PREDICT_D_T
    result_array[inx][1] = pos[1] + vel[1] * constants.PREDICT_D_T


@cuda.jit
def find_density(positions, densities, spatial_lookup, start_indices, result_arrays):
    inx = cuda.grid(1)
    length = len(positions)

    if inx >= length:
        return

    sample_pos = positions[inx]
    density = 0

    nearby_inxs = result_arrays[inx]
    subdivide.get_nearby_particles(inx, positions, spatial_lookup, start_indices, nearby_inxs)

    for i in nearby_inxs:
        if i == length:
            break

        dist = sqrt((positions[i][0] - sample_pos[0])**2 + (positions[i][1] - sample_pos[1])**2)
        
        influence = smoothing.gpu_smoothing_function(dist)

        density += influence * constants.PARTICLE_MASS

    densities[inx] = density
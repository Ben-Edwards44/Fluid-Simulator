import constants
from numba import cuda


@cuda.jit
def colours_from_vel(colours, vels):
    inx = cuda.grid(1)

    if inx >= len(colours):
        return
    
    mag_vel = vels[inx][0]**2 + vels[inx][1]**2
    multiplier = mag_vel / constants.MAX_VEL

    if multiplier > 1:
        multiplier = 1

    red = int(255 * multiplier)
    blue = 255 - red

    colours[inx][0] = red
    colours[inx][1] = 0
    colours[inx][2] = blue


def get_colours(vels, colours):
    vels_d = cuda.to_device(vels)
    colours_d = cuda.to_device(colours)

    num_blocks = (len(colours) + constants.NUM_THREADS - 1) // constants.NUM_THREADS

    colours_from_vel[num_blocks, constants.NUM_THREADS](colours_d, vels_d)

    return colours_d.copy_to_host()
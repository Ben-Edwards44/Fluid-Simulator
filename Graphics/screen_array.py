import constants
from numba import cuda
from numpy import zeros


#move to constants
DRAW_RADIUS = 5


@cuda.jit
def colour_to_bin(colour):
    bin_colour = 0
    for i, x in enumerate(colour):
        bin_colour += x << ((2 - i) * 8)

    return bin_colour


@cuda.jit
def add_to_array(x, y, radius, colour, screen_array):
    if not (0 < x < constants.SCREEN_WIDTH and 0 < y < constants.SCREEN_HEIGHT):
        return
    
    rad_sq = radius * radius

    for i in range(-radius, radius):
        for j in range(-radius, radius):
            dist_sq = i * i + j * j

            if dist_sq < rad_sq:
                if 0 <= x + i < constants.SCREEN_WIDTH and 0 < y + j < constants.SCREEN_HEIGHT:
                    screen_array[x + i][y + j] = colour


@cuda.jit
def draw_circles(positions, colours, screen_array):
    inx = cuda.grid(1)

    if inx >= len(positions):
        return
    
    x, y = positions[inx]
    colour = colour_to_bin(colours[inx])

    add_to_array(x, y, DRAW_RADIUS, colour, screen_array)


def get_screen_array(positions, colours):
    #run on cpu
    screen_array = zeros((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))

    #TODO: reuse device data
    pos_d = cuda.to_device(positions)
    colours_d = cuda.to_device(colours)
    screen_d = cuda.to_device(screen_array)

    num_blocks = (len(positions) + constants.NUM_THREADS - 1) // constants.NUM_THREADS

    draw_circles[num_blocks, constants.NUM_THREADS](pos_d, colours_d, screen_d)

    screen_array = screen_d.copy_to_host()

    return screen_array
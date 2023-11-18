import smoothing
import constants
from numba import cuda
from math import sqrt


@cuda.jit
def apply_gravity(vel):
    vel[1] += constants.GRAVITY


@cuda.jit
def get_mouse_force(mouse_x, mouse_y, pos, vel):
    pos_x, pos_y = pos

    dist = sqrt((mouse_x - pos_x)**2 + (mouse_y - pos_y)**2)

    if dist > constants.MOUSE_INFLUENCE_RADIUS:
        return 0, 0
        
    dir_x = mouse_x - pos_x
    dir_y = mouse_y - pos_y

    strength = 1 - dist / constants.MOUSE_INFLUENCE_RADIUS

    acc_x = dir_x * strength * constants.MOUSE_INFLUENCE
    acc_y = dir_y * strength * constants.MOUSE_INFLUENCE

    return acc_x, acc_y


@cuda.jit
def apply_external_forces(vel, pos, mouse_clicked, mouse_x, mouse_y):
    if constants.APPLY_GRAVITY:
        apply_gravity(vel)

    if mouse_clicked:
        acc_x, acc_y = get_mouse_force(mouse_x, mouse_y, pos, vel)

        vel[0] += acc_x
        vel[1] += acc_y


@cuda.jit
def check_bounds(pos, vel):
    if pos[0] < constants.BOUNDING_LEFT:
        vel[0] *= -constants.VEL_BOUNCE_MUL
        pos[0] = constants.BOUNDING_LEFT
    elif pos[0] > constants.BOUNDING_RIGHT:
        vel[0] *= -constants.VEL_BOUNCE_MUL
        pos[0] = constants.BOUNDING_RIGHT

    if pos[1] < constants.BOUNDING_TOP:
        vel[1] *= -constants.VEL_BOUNCE_MUL
        pos[1] = constants.BOUNDING_TOP
    elif pos[1] > constants.BOUNDING_BOTTOM:
        vel[1] *= -constants.VEL_BOUNCE_MUL
        pos[1] = constants.BOUNDING_BOTTOM


@cuda.jit
def find_pressure(point, positions, nearby_inxs, densities, near_densities, inx, target_density, pressure_multiplier):
    x, y = point

    length = len(positions)

    pressure_x = 0
    pressure_y = 0

    current_density = densities[inx]
    current_near_density = near_densities[inx]

    for i in nearby_inxs:
        if i == inx:
            continue
        elif i == length:
            break

        pos = positions[i]
        dist = sqrt((x - pos[0])**2 + (y - pos[1])**2)

        if dist == 0:
            dist = float((inx + 1) / 100)

        dir_x = (pos[0] - x) / dist
        dir_y = (pos[1] - y) / dist

        pressure_gradient = smoothing.smoothing_derivative(dist)
        near_pressure_gradient = smoothing.near_density_derivative(dist)

        new_density = densities[i]
        new_near_density = near_densities[i]

        shared_pressure, shared_near_pressure = get_shared_pressure(new_density, current_density, current_near_density, new_near_density, target_density, pressure_multiplier)

        #apply pressure
        pressure_x += -shared_pressure * dir_x * pressure_gradient / new_density
        pressure_y += -shared_pressure * dir_y * pressure_gradient / new_density

        #apply near pressure
        pressure_x += shared_near_pressure * dir_x * near_pressure_gradient / new_near_density
        pressure_y += shared_near_pressure * dir_y * near_pressure_gradient / new_near_density

    return pressure_x, pressure_y


@cuda.jit
def find_viscosity(inx, positions, nearby_inxs, vels, visc_strength):
    visc_x = 0
    visc_y = 0

    length = len(positions)

    x, y = positions[inx]

    for i in nearby_inxs:
        if i == length:
            break

        dist = sqrt((positions[i][0] - x)**2 + (positions[i][1] - y)**2)

        if dist <= smoothing.SMOOTHING_RADIUS:
            influence = smoothing.visc_smoothing(dist)

            visc_x += (vels[i][0] - vels[inx][0]) * influence
            visc_y += (vels[i][1] - vels[inx][1]) * influence

    return visc_x * visc_strength, visc_y * visc_strength


@cuda.jit
def apply_viscosity(inx, positions, nearby_inxs, vels, densities, visc_strength):
    visc_x, visc_y = find_viscosity(inx, positions, nearby_inxs, vels, visc_strength)

    acc_x = visc_x / densities[inx]
    acc_y = visc_y / densities[inx]

    vels[inx][0] += acc_x
    vels[inx][1] += acc_y


@cuda.jit
def density_to_pressure(density, near_density, target_density, pressure_multiplier):
    density_error = density - target_density
    pressure = density_error * pressure_multiplier

    near_pressure = near_density * constants.NEAR_PRESSURE_MULTIPLIER

    return pressure, near_pressure


@cuda.jit
def get_shared_pressure(density1, density2, near_density1, near_density2, target_density, pressure_multiplier):
    p1, n1 = density_to_pressure(density1, near_density1, target_density, pressure_multiplier)
    p2, n2 = density_to_pressure(density2, near_density2, target_density, pressure_multiplier)

    shared_pressure = (p1 + p2) / 2
    shared_near_pressure = (n1 + n2) / 2

    return shared_pressure, shared_near_pressure


@cuda.jit
def apply_pressure(positions, nearby_inxs, vels, densities, near_densities, inx, target_density, pressure_multiplier):
    pressure_x, pressure_y = find_pressure(positions[inx], positions, nearby_inxs, densities, near_densities, inx, target_density, pressure_multiplier)

    acc_x = -pressure_x / densities[inx]
    acc_y = -pressure_y / densities[inx]

    vels[inx][0] += acc_x
    vels[inx][1] += acc_y


@cuda.jit
def update_particles(positions, vels, densities, near_densities, nearby_array, target_density, pressure_multiplier, visc_strength, mouse_clicked, mouse_x, mouse_y):
    current_inx = cuda.grid(1)

    if current_inx >= len(positions):
        return
    
    pos = positions[current_inx]
    vel = vels[current_inx]
    nearby_inxs = nearby_array[current_inx]

    apply_pressure(positions, nearby_inxs, vels, densities, near_densities, current_inx, target_density, pressure_multiplier)
    apply_viscosity(current_inx, positions, nearby_inxs, vels, densities, visc_strength)
    check_bounds(pos, vel)
    apply_external_forces(vel, pos, mouse_clicked, mouse_x, mouse_y)


@cuda.jit
def update_pos_from_vel(positions, vels, delta_time):
    inx = cuda.grid(1)

    if inx >= len(positions):
        return
    
    positions[inx][0] += vels[inx][0] * delta_time
    positions[inx][1] += vels[inx][1] * delta_time
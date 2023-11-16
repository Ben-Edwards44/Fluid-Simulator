class Particle:
    def __init__(self, pos, colour, array_inx):
        self.x, self.y = pos
        self.colour = colour
        self.array_inx = array_inx

        self.vel_x = 0
        self.vel_y = 0

    def update(self, positions, vels, colours):
        self.x, self.y = positions[self.array_inx]
        self.vel_x, self.vel_y = vels[self.array_inx]
        self.colour = colours[self.array_inx]

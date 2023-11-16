import pygame_ui_toolkit
from Physics import constants
import smoothing


SLIDER_X = 100
LENGTH = 100


init = False


def create_sliders(window):
    global init
    global target_density
    global pressure_multiplier
    #global smoothing_radius
    
    init = True

    target_density = pygame_ui_toolkit.slider_value_text.create_slider_text(window, SLIDER_X, 50, SLIDER_X + LENGTH // 2 + 30, 50, LENGTH, 5, 0, 30, 18, (255, 255, 255), None, 32, (255, 255, 255))
    pressure_multiplier = pygame_ui_toolkit.slider_value_text.create_slider_text(window, SLIDER_X, 100, SLIDER_X + LENGTH // 2 + 30, 100, LENGTH, 5, 0, 40, 5, (255, 255, 255), None, 32, (255, 255, 255))
    #smoothing_radius = pygame_ui_toolkit.slider_value_text.create_slider_text(window, SLIDER_X, 150, SLIDER_X + LENGTH // 2 + 30, 150, LENGTH, 5, 0, 2, 1.6, (255, 255, 255), None, 32, (255, 255, 255))


def update():
    target_density.update()
    pressure_multiplier.update()
    #smoothing_radius.update()

    pygame_ui_toolkit.slider_value_text.blit_slider_text(target_density)
    pygame_ui_toolkit.slider_value_text.blit_slider_text(pressure_multiplier)
    #pygame_ui_toolkit.slider_value_text.blit_slider_text(smoothing_radius)

    constants.target_density = target_density.value
    constants.pressure_multiplier = pressure_multiplier.value
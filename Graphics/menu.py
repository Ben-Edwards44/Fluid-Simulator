import constants
import pygame_ui_toolkit


init = False


def create_elements(window):
    global init
    global restart
    global target_density, pressure_multiplier, visc_strength
    
    init = True

    target_density = pygame_ui_toolkit.slider_value_text.create_slider_text(window, constants.SLIDER_X, 50, constants.SLIDER_X + constants.LENGTH // 2 + 30, 50, constants.LENGTH, 5, 0, 30, 18, (255, 255, 255), None, 32, (255, 255, 255))
    pressure_multiplier = pygame_ui_toolkit.slider_value_text.create_slider_text(window, constants.SLIDER_X, 100, constants.SLIDER_X + constants.LENGTH // 2 + 30, 100, constants.LENGTH, 5, 0, 40, 5, (255, 255, 255), None, 32, (255, 255, 255))
    visc_strength = pygame_ui_toolkit.slider_value_text.create_slider_text(window, constants.SLIDER_X, 150, constants.SLIDER_X + constants.LENGTH // 2 + 30, 150, constants.LENGTH, 5, 0, 1, 0.3, (255, 255, 255), None, 32, (255, 255, 255))
    
    restart_btn = pygame_ui_toolkit.button.RectButton(window, 1100, 50, (255, 255, 255), 100, 30, corner_radius=4)
    pygame_ui_toolkit.button_size_change.change_existing_button(restart_btn, (100, 30), (120, 40), (80, 20))
    restart = pygame_ui_toolkit.button.TextWrapper(restart_btn, "Restart", (0, 0, 0), 32)


def update():
    target_density.update()
    pressure_multiplier.update()
    visc_strength.update()
    restart.update()

    pygame_ui_toolkit.slider_value_text.blit_slider_text(target_density)
    pygame_ui_toolkit.slider_value_text.blit_slider_text(pressure_multiplier)
    pygame_ui_toolkit.slider_value_text.blit_slider_text(visc_strength)

    constants.target_density = target_density.value
    constants.pressure_multiplier = pressure_multiplier.value
    constants.visc_strength = visc_strength.value

    return restart.button_object.clicked
import vizdoom
import model
import classes
import time
import numpy as np
import matplotlib.pyplot as plt

cnn = classes.Model()
game = vizdoom.DoomGame()
game.load_config("gamedata/training_maps/config_training.cfg")
game.set_window_visible(False)
game.init()
game.new_episode()
for a in model.actions :
    if not game.is_episode_finished() :
        state = game.get_state()
        img = state.screen_buffer
        cnn.forward(img)
        screen  = np.moveaxis(state.screen_buffer,0, 2)
        plt.imshow(screen)
        plt.show()
        game.make_action(a)
    time.sleep(0.01)
game.close()

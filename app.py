import vizdoom
import model
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

cnn = model.Model()
game = vizdoom.DoomGame()
game.load_config("gamedata/training_maps/config_training.cfg")
game.set_window_visible(False)
game.init()
game.new_episode()
if not game.is_episode_finished() :
    state = game.get_state()
    # img = np.expand_dims(state.screen_buffer, axis=0)
    # img = torch.from_numpy(img).float()
    img = torch.zeros([1, 4, 320, 200])
    cnn.forward(img)
time.sleep(0.01)
game.close()

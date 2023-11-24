# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:58:17 2023

@author: Gabriel
"""

import os
import matplotlib.pyplot as plt
import numpy as np

import vizdoom
# from vizdoom import *
import random
import time

game = vizdoom.DoomGame()
# game.load_config(os.path.join(vizdoom.scenarios_path,"basic.cfg"))
game.load_config("C:/Users/Gabriel/Documents/_Documents/Etudes/3A 2/9.6.1 deep learning and RL/projet/custom map/config1.cfg")

game.set_window_visible(False)


game.init()

shoot = [0, 0, 1]
left = [1, 0, 0]
right = [0, 1, 0]
actions = [shoot, left, right]

episodes = 10
for i in range(episodes):
    game.new_episode()
    t=0
    while not game.is_episode_finished():
        state = game.get_state()
        img = state.screen_buffer
        misc = state.game_variables
        # reward = game.make_action(random.choice(actions))
        reward = game.make_action([1,0,1])
        screen  = np.moveaxis(state.screen_buffer,0, 2)
        plt.imshow(screen)
        if t==20:
            # save
            pass
        plt.show()
        print("\treward:", reward)
        time.sleep(0.02)
        t+=1
    print("Result:", game.get_total_reward())
    time.sleep(2)
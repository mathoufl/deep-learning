# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:58:17 2023

@author: Gabriel
"""

import matplotlib.pyplot as plt
import numpy as np

import vizdoom

import cv2

from utils import demoToGIF, demoToMP4, convertActionIdToButtonsArray


game = vizdoom.DoomGame()

game.load_config( "gamedata/training_maps/config_training.cfg" )

# game.set_episode_timeout(50)


maps = ["MAP01", "MAP02"]

game.set_window_visible(True)

# game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
game.init()


episodes = 5
for i in range(episodes):
    # we choose one of the map at random
    game.set_doom_map(np.random.choice(maps))
    game.new_episode()
    t=0
    while not game.is_episode_finished():
        state = game.get_state()
        img = state.screen_buffer
        misc = state.game_variables

        
        # available_buttons =
        #       ATTACK
        # 		SPEED
        # 		MOVE_RIGHT
        # 		MOVE_LEFT
        # 		MOVE_BACKWARD
        # 		MOVE_FORWARD
        # 		TURN_RIGHT
        # 		TURN_LEFT
        random_action = convertActionIdToButtonsArray(np.random.randint(108),[2,2,3,3,3])
        reward = game.make_action(random_action)
        # screen  = np.moveaxis(state.screen_buffer,0, 2)
        # plt.imshow(screen)
        # plt.imshow(img, cmap="gray")

        # plt.show()
        # print("\treward:", reward)
        t+=1
    
    print("Result:", game.get_total_reward())
game.close()


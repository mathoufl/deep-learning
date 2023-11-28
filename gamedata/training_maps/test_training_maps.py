# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:58:17 2023

@author: Gabriel
"""

import matplotlib.pyplot as plt
import numpy as np

import vizdoom


# game.load_config(os.path.join(vizdoom.scenarios_path,"basic.cfg"))
path_maps = "C:/Users/Gabriel/Documents/_Documents/Etudes/3A 2/9.6.1 deep learning and RL/projet/custom map/training maps/"
# configs = [path_maps+"config_map_1.cfg",path_maps+"config_map_2.cfg"]

game = vizdoom.DoomGame()

game.load_config( path_maps + "config_training.cfg" )

game.set_episode_timeout(50)


maps = ["MAP01", "MAP02"]

# game.set_window_visible(False)


game.init()


episodes = 10
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
        reward = game.make_action([1,0,1,0,0,1,1,0])
        # screen  = np.moveaxis(state.screen_buffer,0, 2)
        # plt.imshow(screen)
        plt.imshow(img, cmap="gray")

        plt.show()
        print("\treward:", reward)
        t+=1
    
    print("Result:", game.get_total_reward())
game.close()
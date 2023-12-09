# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:54:43 2023

@author: Gabriel
"""

import torch
import vizdoom
import os
import numpy as np
from collections import deque
import copy
import datetime

from model import Model, BasicModel
from training import train_model, test_model

import cv2
import imageio

from utils import convertActionIdToButtonsArray

import time


def watch_model(model, env_config_file, save_demo=False, window_visible=False, 
                n_imagesByState = 4, real_time=False, color=False,
                img_dim=(240,320), action_signature=[2,2,3,3,3], 
                n_episodes=10, frame_skip=0, doom_map_list=["MAP01", "MAP02"], device = "cuda",
                video_name=None,
                dtype = torch.float):
    game = vizdoom.DoomGame()
    model.eval()
    # we will store the demo in the folder :
    d = datetime.datetime.now()
    date = f"{d.day:02}-{d.month:02}-{d.year} - {d.hour:02}h{d.minute:02}min{d.second:02}s"
    os.makedirs("test_demos/test_demos_"+date)
    
    game.load_config(env_config_file)
    if window_visible:
        game.set_window_visible(True)
    game.init()
    
    video = None
    save_mp4 = video_name is not None
        
    
    rewards_history = []
    
    for i_episode in range(n_episodes):
        
        if save_mp4:
            video_file_name = video_name + f"#{i_episode}.mp4"
            video = cv2.VideoWriter(video_file_name,
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    35, (img_dim[1],img_dim[0]), isColor = True)
        
        game.set_doom_map(np.random.choice(doom_map_list))
        # here we specify a file to store the demo to if we want
        # game.new_episode(dir_training_report + "/episode_demos/episode#"+str(i_episode))
        if save_demo:
            game.new_episode("test_demos/test_demos_"+date+"/episode#{i_episode:04}")
        else:
            game.new_episode()
        
        # initialize the image buffer with n_imagesByState blank image (in gray scale)
        # this is wath the agent will see : the n last images of the environnement
        image_buffer = deque([], n_imagesByState)
        # image_buffer = deque(np.asarray(np.zeros([img_dim[0],img_dim[1],n_imagesByState])).tolist(),
        #                      n_imagesByState)
        
        # fill the image buffer with the first few frames of the episode
        for i in range(n_imagesByState):
            image_buffer.append(game.get_state().screen_buffer)
            
            
        state_image = None
        
        while not game.is_episode_finished():
            # the env automatically finish after a number of step definied in the config file
            # using only one stop condition for this loop is fine
            if real_time and window_visible:
                time.sleep(1/35)
            state_image = np.array(image_buffer)
            
            
            # greedy action :
           
            with torch.no_grad():
                # the different images are different channels for the model
                # the channels are the first dimension after the batch dimesnsion in conv2D
                state_tensor = torch.tensor(state_image, device=device, dtype=dtype)
                # state_tensor = state_tensor.unsqueeze(0)
                q_values = model(state_tensor)[2]
                action_id = torch.argmax(q_values).cpu().item()    
            action = convertActionIdToButtonsArray(action_id, action_signature)
            # --------------------------------
            
            # the agent will hold this action for frame_skip+1 ticks, but still "watch" what's happening
            for _ in range(frame_skip+1):
                game.make_action(action)
                if not game.is_episode_finished():
                    image_buffer.append(game.get_state().screen_buffer)
                    if save_mp4:
                        img = np.moveaxis(game.get_state().screen_buffer,0, 2)
                        video.write( cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else: # if the episode end during this time 
                    image_buffer.append(list(np.zeros(img_dim)))
        rewards_history.append(game.get_total_reward())
        if save_mp4:
            video.release()
            cv2.destroyAllWindows()
    game.close()
    return rewards_history



#%%
path_model = r"C:\Users\Gabriel\Documents\_Documents\Etudes\3A 2\9.6.1 deep learning and RL\projet\saved_model\training_report_07-12-2023---11h00min35s\model#94"



model = torch.load(r"C:\Users\Gabriel\Documents\_Documents\Etudes\3A 2\9.6.1 deep learning and RL\projet\saved_model" +
                   "\\model#574",
                   map_location="cuda")
model = torch.load(r"C:\Users\Gabriel\Documents\_Documents\Etudes\3A 2\9.6.1 deep learning and RL\projet\saved_model" +
                   "\\model850+#569",
                   map_location="cuda")
# model = BasicModel().to("cuda")

rewards=test_model(model,
            env_config_file="gamedata/training_maps/fire_and_dodge_color.cfg",
            img_dim=(120,160),
            n_imagesByState = 1,
            color=True,
            frame_skip=0,
            action_signature=[2,3],
 
            n_episodes=1,
            window_visible = True,
            real_time = False,
            save_demo = False,
            doom_map_list=["FIR_DODG"],
            save_mp4 = True,
            video_name = "vid/model#850+569_test"
            )
print(rewards)
# demo_file=r"C:\Users\Gabriel\Documents\_Documents\Etudes\3A 2\9.6.1 deep learning and RL\projet\saved_model\training_report_07-12-2023---11h00min35s\demo\episode#0200.lmp"

# demoToMP4("demo_training.mp4", demo_file, "gamedata/training_maps/fire_and_dodge.cfg",
#               resolution=vizdoom.ScreenResolution.RES_640X480)



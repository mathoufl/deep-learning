import numpy as np
import torch
from collections import deque

import vizdoom

from model import Model
from utils import convertActionIdToButtonsArray


"""
 take a model and train it
 
 for now this version only supports grayscale images
"""

def train_model(model, n_imagesByState, img_dim, action_dim, env_config_file, n_episodes,
                replay_buffer_size, epsilon_schedule, frame_skip=0, doom_map_list=["MAP01", "MAP02"], device = "cuda"):
    
    target_model = model.deepcopy() # not the actual synthax
    # we make a copy of the model that will be used for calculating the target
    # this increase stability
    # it will follow the actual model with some lag
    
    
    
    game = vizdoom.DoomGame()
    

    game.load_config(env_config_file)
    maps = ["MAP01", "MAP02"]
    
    for i_episode in range(n_episodes):
        
        epsilon = epsilon_schedule(i_episode)
        game.set_doom_map(np.random.choice(doom_map_list))
        game.new_episode()
        
        # initialize the image buffer with n_imagesByState blank image (in gray scale)
        # this is wath the agent will see : the n last images of the environnement
        image_buffer = deque([], n_imagesByState)
        # image_buffer = deque(np.asarray(np.zeros([img_dim[0],img_dim[1],n_imagesByState])).tolist(),
        #                      n_imagesByState)
        
        # fill the image buffer with the first few frames of the episode
        for i in range(n_imagesByState):
            image_buffer.append(game.get_state().screen_buffer)
            
        state_image, previous_state_image = None, None
        
        while not game.is_episode_finished():
            # the env automatically finish after a number of step definied in the config file
            # using only one stop condition for this loop is fine
            
    
            previous_state_image = state_image
            state_image = torch.tensor(image_buffer, device=device)
            # the different images are different channels for the model
            # the channels are the first dimension after the batch dimesnsion in conv2D
            torch.movedim(state_image, 2, 0) 
            state_image = state_image.unsqueeze(0) # made it a batch of 1 (standard practice in pytorch)
            
            
            # ---- epsilon greedy policy ----
            action_id = None
            if np.random.random() < epsilon:
                # exploratory action :
                action_id = np.random.random(action_dim)
            else:
                # greedy action :
                with torch.no_grad():
                    
                    q_values = model(state_image)
                    action_id = torch.argmax(q_values).item()
                    
            action = convertActionIdToButtonsArray(action_id) # action signature! we are using the default one here
            # --------------------------------
            
            # the agent will hold this action for frame_skip+1 ticks
            reward = 0
            for _ in range(frame_skip+1):
                reward += game.make_action(action)
                image_buffer.append(game.get_state().screen_buffer)
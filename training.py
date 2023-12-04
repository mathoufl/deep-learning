import os
import numpy as np
import torch
from collections import deque
import copy
import datetime

import vizdoom

from model import Model
from utils import convertActionIdToButtonsArray

from learn import learn


import matplotlib.pyplot as plt

class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.cursor = 0 # this is where to had the next element

    def store_transition(self, state, action, reward, next_state, non_final):
        if len(self.memory) < self.capacity:
            self.memory.append([state, action, reward, next_state, non_final])
        else:
            self.memory[self.cursor] = [state, action, reward, next_state, non_final]
        self.cursor = (1 + self.cursor) % self.capacity

    def sample_batches(self, batch_size, device="cuda"):
         idx = np.random.choice([i for i in range(len(self.memory))], batch_size, replace=False).tolist()
         # sample = [self.memory[i] for i in idx]
         batch_state      = []
         batch_reward     = []
         batch_action     = []
         batch_next_state = []
         batch_non_final  = []
         for i in idx:
             batch_state.append(self.memory[i][0])
             batch_reward.append(self.memory[i][1])      
             batch_action.append(self.memory[i][2])
             batch_next_state.append(self.memory[i][3])
             batch_non_final.append(self.memory[i][4])
         batch_state      = torch.tensor(  np.array(batch_state)  , device = device, dtype = torch.float)
         batch_reward     = torch.tensor(  batch_reward  , device = device, dtype = torch.float)
         batch_action     = torch.tensor(  batch_action  , device = device, dtype = torch.int64)
         batch_next_state = torch.tensor(  batch_next_state  , device = device, dtype = torch.float)
         batch_non_final  = torch.tensor(  batch_non_final  , device = device, dtype = torch.float)
             
         
         return batch_state, batch_reward, batch_action, batch_next_state, batch_non_final

    def __len__(self):
        return len(self.memory)

"""
 take a model and train it
 
 for now this version only supports grayscale images
"""

def train_model(model, env_config_file, n_imagesByState = 4,
                img_dim=(240,320), action_signature=[2,2,3,3,3], 
                n_episodes=100,
                epsilon_schedule = (lambda t : 0.1),
                replay_buffer_capacity=500,
                batch_size = 16, learning_rate =1e-4, weight_decay=5e-3,
                frame_skip=0, doom_map_list=["MAP01", "MAP02"], device = "cuda"):
    
    target_model = copy.deepcopy(model)
    # we make a copy of the model that will be used for calculating the target
    # this increase stability
    # it will follow the actual model with some lag
    
    optim =  torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay )
    replay_buffer = ReplayBuffer(replay_buffer_capacity)
    
    action_dim = np.prod(action_signature)
    
    game = vizdoom.DoomGame()
    
    # we will store the training things in the folder :
    d = datetime.datetime.now()
    date = f"{d.day:02}-{d.month:02}-{d.year} - {d.hour:02}h{d.minute:02}min{d.second:02}s"
    dir_training_report = "../training_reports/training report " + date
    os.mkdir( dir_training_report)
    os.mkdir(dir_training_report + "/episode_demos")
    os.mkdir(dir_training_report + "/model_checkpoints")
    
    game.load_config(env_config_file)
    game.set_window_visible(True)
    game.init()
    
    for i_episode in range(n_episodes):
        
        epsilon = epsilon_schedule(i_episode)
        game.set_doom_map(np.random.choice(doom_map_list))
        # here we specify a file to store the demo to if we want
        # game.new_episode(dir_training_report + "/episode_demos/episode#"+str(i_episode))
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
        
        qvalues_during_training = []
        
        while not game.is_episode_finished():
            # the env automatically finish after a number of step definied in the config file
            # using only one stop condition for this loop is fine
            
            state_image = np.array(image_buffer)

            
            # ---- epsilon greedy policy ----
            action_id = None
            if np.random.random() < epsilon:
                # exploratory action :
                action_id = np.random.randint(action_dim)
            else:
                # greedy action :
                with torch.no_grad():
                    # the different images are different channels for the model
                    # the channels are the first dimension after the batch dimesnsion in conv2D
                    state_tensor = torch.tensor(state_image, device=device, dtype=torch.float)
                    # torch.movedim(state_tensor, 2, 0)  # it is useless !?
                    state_tensor = state_tensor.unsqueeze(0)
                    q_values = model(state_tensor)[2]
                    action_id = torch.argmax(q_values).cpu().item()
                    
                    qvalues_during_training.append( torch.max(q_values).cpu().item() )       
            action = convertActionIdToButtonsArray(action_id, action_signature)
            # --------------------------------
            
            # the agent will hold this action for frame_skip+1 ticks, but still "watch" what's happening
            reward = 0
            for _ in range(frame_skip+1):
                reward += game.make_action(action)
                
                # plt.imshow(game.get_state().screen_buffer, cmap="gray")
                # plt.show()
                
                if not game.is_episode_finished():
                    image_buffer.append(game.get_state().screen_buffer)
                else:
                    image_buffer.append(list(np.zeros(img_dim)))
                # what if the episode end during this time ?
                
            # ----- storing the transition in the replay buffer ------
            non_final = not game.is_episode_finished()
            next_state = np.array(image_buffer) # this is the frames the agent observed while holding the action
            
            replay_buffer.store_transition(state_image, action_id, reward, next_state, non_final)
            
            # LEARNING TIME - we lean after the end of each episode
            if len(replay_buffer) >= batch_size*4:
                batch_state, batch_action, batch_reward, batch_next_state, batch_non_final =\
                                                             replay_buffer.sample_batches(batch_size)
                learn(model, target_model, optim,
                      batch_state, batch_action, batch_reward, batch_next_state, batch_non_final)
                
        # end of the episode
        print("\nepisode#"+str(i_episode)+"finished !")
        
        # we update the target network at the end of each episode 
        target_model.load_state_dict(model.state_dict())
        
        total_reward = game.get_total_reward()
        average_q = np.mean(qvalues_during_training)
        print("total training reward : ", total_reward)
        print("average training max q value : ", average_q, "\n")
        
        # test step :
        test_rewards = []
        for doom_map in doom_map_list:
            rewards = test_model(model, env_config_file, doom_map_list=[doom_map], n_episodes=10,
                                n_imagesByState = n_imagesByState,
                                img_dim=img_dim, action_signature=action_signature, 
                                frame_skip=frame_skip,device = device)
            test_rewards.append(np.mean(rewards))
            print("average test rewards on "+doom_map+f" : {test_rewards[-1]}")
        model.train()
        
        # store some training tracking metrics
        reward_file = open(dir_training_report+"/training_rewards.txt", "a") 
        reward_file.write(str(total_reward)+"\n")
        reward_file.close()
        test_reward_file = open(dir_training_report+"/testing_rewards.txt", "a") 
        test_reward_file.write(str(test_rewards)+"\n")
        test_reward_file.close()
        average_q_file = open(dir_training_report+"/average_q.txt", "a") 
        average_q_file.write(str(average_q)+"\n")
        
        average_q_file.close()
        
        if (i_episode+1) % 5 == 0 or i_episode == n_episodes-1:
            torch.save(model, dir_training_report + "/model_checkpoints/model#"+str(i_episode))
        
        

def test_model(model, env_config_file, save_demo=False, window_visible=False, n_imagesByState = 4,
                img_dim=(240,320), action_signature=[2,2,3,3,3], 
                n_episodes=10, frame_skip=0, doom_map_list=["MAP01", "MAP02"], device = "cuda"):
    game = vizdoom.DoomGame()
    model.eval()
    # we will store the demo in the folder :
    d = datetime.datetime.now()
    date = f"{d.day:02}-{d.month:02}-{d.year} - {d.hour:02}h{d.minute:02}min{d.second:02}s"
    os.mkdir("test_demos")
    os.mkdir("test_demos/test_demos_"+date)
    
    game.load_config(env_config_file)
    if window_visible:
        game.set_window_visible(True)
    game.init()
    
    rewards_history = []
    
    for i_episode in range(n_episodes):
        
        
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
            
            state_image = np.array(image_buffer)
            
            # greedy action :
            with torch.no_grad():
                # the different images are different channels for the model
                # the channels are the first dimension after the batch dimesnsion in conv2D
                state_tensor = torch.tensor(state_image, device=device, dtype=torch.float)
                state_tensor = state_tensor.unsqueeze(0)
                q_values = model(state_tensor)[2]
                action_id = torch.argmax(q_values).cpu().item()    
            action = convertActionIdToButtonsArray(action_id, action_signature)
            # --------------------------------
            
            # the agent will hold this action for frame_skip+1 ticks, but still "watch" what's happening
            for _ in range(frame_skip+1):
                game.make_action(action)
                if not game.is_episode_finished():
                    image_buffer.append(game.get_state().screen_buffer)
                else: # if the episode end during this time 
                    image_buffer.append(list(np.zeros(img_dim)))
        rewards_history.append(game.get_total_reward())
    return rewards_history
                

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

import time



class ReplayBuffer(object):

    def __init__(self, capacity, dtype=torch.float):
        self.capacity = capacity
        self.dtype = dtype
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
         dtype = self.dtype
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
         batch_state      = torch.tensor( np.array(batch_state)  , device = device, dtype = dtype)
         batch_reward     = torch.tensor(  batch_reward  , device = device, dtype = dtype)
         batch_action     = torch.tensor(  batch_action  , device = device, dtype = torch.int64)
         batch_next_state = torch.tensor(  np.array(batch_next_state)  , device = device, dtype = dtype)
         batch_non_final  = torch.tensor(  batch_non_final  , device = device, dtype = dtype)
         
         return batch_state, batch_reward, batch_action, batch_next_state, batch_non_final

    def __len__(self):
        return len(self.memory)

"""
 take a model and train it
 
 for now this version only supports grayscale images
"""

def train_model(model, env_config_file, training_report_dir="../training_reports",
                n_imagesByState = 4,
                img_dim=(240,320), action_signature=[2,2,3,3,3], color=False,
                n_episodes=100, window_visible=False, save_demo=True,
                target_update_freq=32,
                epsilon_schedule = (lambda t : 0.1),
                replay_buffer_capacity=500,
                batch_size = 16, learning_rate =1e-4, weight_decay=5e-3,
                early_stopping_score = [torch.inf,torch.inf],
                frame_skip=0, doom_map_list=["MAP01", "MAP02"], device = "cuda",
                dtype=torch.float):
    
    target_model = copy.deepcopy(model)
    # we make a copy of the model that will be used for calculating the target
    # this increase stability
    # it will follow the actual model with some lag
    
    optim =  torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay )
    replay_buffer = ReplayBuffer(replay_buffer_capacity, dtype)
    
    action_dim = np.prod(action_signature)
    
    game = vizdoom.DoomGame()
    
    # we will store the training things in the folder :
    d = datetime.datetime.now()
    date = f"{d.day:02}-{d.month:02}-{d.year} - {d.hour:02}h{d.minute:02}min{d.second:02}s"
    dir_training_report =  training_report_dir + "/training report " + date
    os.mkdir( dir_training_report)
    os.mkdir(dir_training_report + "/episode_demos")
    os.mkdir(dir_training_report + "/model_checkpoints")
    
    game.load_config(env_config_file)
    if window_visible:
        game.set_window_visible(True)
    game.init()
    
    clock_target_update = 0 # tell us when to update the target network
    for i_episode in range(n_episodes):
        
        epsilon = epsilon_schedule(i_episode)
        game.set_doom_map(np.random.choice(doom_map_list))
        # here we specify a file to store the demo to if we want
        if save_demo:
            
            game.new_episode(dir_training_report + f"/episode_demos/episode#{i_episode:04}")
        else:
            game.new_episode()
        
        # initialize the image buffer with n_imagesByState blank image (in gray scale)
        # this is wath the agent will see : the n last images of the environnement
        image_buffer = deque([], n_imagesByState)
        # image_buffer = deque(np.asarray(np.zeros([img_dim[0],img_dim[1],n_imagesByState])).tolist(),
        #                      n_imagesByState)
        
        # fill the image buffer with the first few frames of the episode
        for i in range(n_imagesByState):
            image_buffer.append(np.array(game.get_state().screen_buffer))
            
        state_image = None
        qvalues_during_training = []
        training_losses = []
        
        while not game.is_episode_finished():
            # the env automatically finish after a number of step definied in the config file
            # using only one stop condition for this loop is fine
            
            if color:
                state_image = np.array(image_buffer).reshape(3*n_imagesByState,img_dim[0],img_dim[1])
            else:
                state_image = np.array(image_buffer).reshape(1*n_imagesByState,img_dim[0],img_dim[1])

            
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
                    state_tensor = torch.tensor(state_image, device=device, dtype=dtype)
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
                    if color:
                        image_buffer.append(np.zeros( (3,img_dim[0],img_dim[1]) ))
                    else:
                        image_buffer.append(np.zeros(img_dim))
                # what if the episode end during this time ?
                
            # ----- storing the transition in the replay buffer ------
            non_final = not game.is_episode_finished()
            # this is the frames the agent observed while holding the action
            if color:
                next_state = np.array(image_buffer).reshape(3*n_imagesByState,img_dim[0],img_dim[1])
            else:
                next_state = np.array(image_buffer).reshape(1*n_imagesByState,img_dim[0],img_dim[1])
            
            replay_buffer.store_transition(state_image, action_id, reward, next_state, non_final)
            
            # LEARNING TIME - we lean after each step
            # print(f"optimization step#{clock_target_update} :")
            # t_ = time.time()
            if len(replay_buffer) >= batch_size*4:
                batch_state, batch_action, batch_reward, batch_next_state, batch_non_final =\
                                                             replay_buffer.sample_batches(batch_size)
                training_losses.append( learn(model, target_model, optim,
                      batch_state, batch_action, batch_reward, batch_next_state, batch_non_final) )
                
                clock_target_update+=1
                if clock_target_update % target_update_freq ==0:
                    # we update the target network once every target_update_freq training steps
                    target_model.load_state_dict(model.state_dict())
            # print(f"optimisation duration : {(time.time()-t_)*1000:.7}ms")
        # end of the episode
        print("\nepisode#"+str(i_episode)+"finished !")
        
        
        
        total_reward = game.get_total_reward()
        average_q = np.mean(qvalues_during_training)
        average_loss = np.mean(training_losses)
        print("total training reward : ", total_reward)
        print("average training max q value : ", average_q)
        print("average training loss : ", average_loss)
        
        # test step :
        test_rewards = []
        early_stop = True
        for i in range(len(doom_map_list)):
            doom_map = doom_map_list[i]
            rewards = test_model(model, env_config_file, doom_map_list=[doom_map], n_episodes=5,
                                n_imagesByState = n_imagesByState,
                                img_dim=img_dim, action_signature=action_signature, 
                                frame_skip=frame_skip,device = device, dtype = dtype)
            test_rewards.append(np.mean(rewards))
            early_stop = early_stop and (np.mean(rewards) >= early_stopping_score[i] ) 
            print("average test rewards on "+doom_map+f" : {test_rewards[-1]}")
        print("")
        
        model.train()
        
        # store some training tracking metrics
        reward_file = open(dir_training_report+"/training_rewards.txt", "a") 
        reward_file.write(str(total_reward)+"\n")
        reward_file.close()
        loss_file = open(dir_training_report+"/training_loss.txt", "a") 
        loss_file.write(str(average_loss)+"\n")
        loss_file.close()
        test_reward_file = open(dir_training_report+"/testing_rewards.txt", "a") 
        test_reward_file.write(str(test_rewards)+"\n")
        test_reward_file.close()
        average_q_file = open(dir_training_report+"/average_q.txt", "a") 
        average_q_file.write(str(average_q)+"\n")
        
        average_q_file.close()
        
        if (i_episode+1) % 5 == 0 or i_episode == n_episodes-1 or early_stop:
            torch.save(model, dir_training_report + "/model_checkpoints/model#"+str(i_episode))
        if early_stop:
            print("early stop")
            break
        
        

def test_model(model, env_config_file, save_demo=False, window_visible=False, 
                n_imagesByState = 4, real_time=False, color=False,
                img_dim=(240,320), action_signature=[2,2,3,3,3], 
                n_episodes=10, frame_skip=0, doom_map_list=["MAP01", "MAP02"], device = "cuda",
                save_mp4 = False, video_name="video",
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
    if save_mp4:
        import cv2
    
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
                

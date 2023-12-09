
from model import Model, BasicModel, Model_vgg
from training import train_model
from utils import demoToMP4, demoToGIF

import torch

# Now with a VGG moodel and colors

#load the last version of the model
model = torch.load("/home/docker/training/source3/training_reports/training report 07-12-2023 - 18h28min52s/model_checkpoints/model#249",
    map_location="cuda")

def epsilon_sch(t):
    plateau_d = 0
    n_ep = 600
    e_start, e_end = 0.3,0.05
    if t<plateau_d:
        return 1.0
    else:
        return max(e_start - (t-plateau_d)*(e_start-e_end)/(n_ep-plateau_d) , e_end)


#model.half()

import numpy as np
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("nbr params :", params)

train_model(model,
            env_config_file="gamedata/training_maps/fire_and_dodge_color.cfg",
            training_report_dir="/home/docker/training/source3/training_reports",
            img_dim=(120,160),
            n_imagesByState = 1,
            color=True,
            frame_skip=0,
            action_signature=[2,3],
            replay_buffer_capacity=10000,
            batch_size = 32,
            learning_rate =8e-5,
            weight_decay=1e-5,
            n_episodes=600,
            early_stopping_score=[200],
            epsilon_schedule=epsilon_sch,
            window_visible = False,
            save_demo = False,
            doom_map_list=["FIR_DODG"],
            dtype = torch.float
            )
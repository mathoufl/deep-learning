# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 16:38:28 2023

@author: Gabriel
"""

from model import Model
from training import train_model

model = Model()
model.to("cuda")

train_model(model, env_config_file="gamedata/training_maps/config_training.cfg")
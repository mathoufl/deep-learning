# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 14:16:11 2023

@author: Gabriel
"""

import sys
import os

from utils import demoToMP4

if __name__ == "__main__":
    print(sys.argv[1])
    vid_name = os.path.split(sys.argv[1])[1] + ".mp4"
    
    demoToMP4(vid_name, sys.argv[1], "gamedata/training_maps/config_training.cfg")
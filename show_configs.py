# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 11:41:16 2022

@author: A00006846

"""
import hypertune
import json
import copy
import sys
cfg = {}
file_name = "cfg/temp5.json"
if (len(sys.argv))>1:
    file_name = sys.argv[1]

with open(file_name, "r") as jfile:
    cfg = json.load(jfile)

cfgs = hypertune.generate_configs(cfg)
[print(x,"\n\n\n\n") for x in cfgs]



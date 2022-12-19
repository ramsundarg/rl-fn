# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 11:41:16 2022

@author: A00006846

"""
import numpy as np
import json
import copy
import os

import run_experiment

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


pidx = 0
pitems = []
config_count = 0
def hypertune(level, idx, hypertune_items):
    global pidx
    global pitems,config_count,cfg_copy,cfg
    if idx == len(hypertune_items):
        if level == 0:
            cfg_copy.pop('tune', None)
            run_experiment.run_experiment(cfg_copy)
            
            print(cfg_copy)
            config_count= config_count+1
            print(config_count)
            print("\n\n\n")
            
        else:
            hypertune(level-1, pidx+1, pitems)
        
        return

    
    if (list(hypertune_items)[idx]) == "group":
        for el in list(hypertune_items.values())[idx]:
            pidx = idx
            pitems = hypertune_items
            hypertune(level+1, 0, el)
            for d in el.keys():
                keys =d.split('.')
                c = cfg_copy
                co = cfg
                update_key = True
                for key in keys[0:-1]:
                    if co.get(key, None) == None:
                        del c[key] 
                        update_key = False
                        break
                    if c.get(key, None) == None:
                        c[key] = []
                    co =co[key]
                    c= c[key]
                if update_key:
                    c[keys[-1]]= co[keys[-1]]
                
        return
    keys = (list(hypertune_items)[idx]).split('.')

    d = cfg_copy
    for key in keys[0:-1]:
        if d.get(key, None) == None:
            d[key] = []
        d = d.get(key)
    tune_item = list(hypertune_items.values())[idx]
    if(tune_item.get("low", "") != ""):
        is_value = False
        # print(tune_item["low"],tune_item["high"],tune_item["step"])
        for val in np.arange(tune_item["low"], tune_item["high"], tune_item["step"]):
            d[keys[-1]] = val.item()
            hypertune(level, idx+1, hypertune_items)
    if(tune_item.get("set")):
        for val in tune_item["set"]:
            d[keys[-1]] = eval(val)
            hypertune(level, idx+1, hypertune_items)
    if(tune_item.get("list")):
        for val in tune_item["list"]:
            d[keys[-1]] = val
            hypertune(level, idx+1, hypertune_items)
        



cfg = {}

file_name = "cfg/temp4.json"
with open(file_name, "r") as jfile:
    cfg = json.load(jfile)

cfg_copy = copy.deepcopy(cfg)
hypertune(0, 0, cfg_copy.get('tune', []))




import numpy as np
import json
import os
import multiprocessing
import run_experiment
import copy
"""
The hypertunable configuration framework implementation.
"""
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


pidx = 0
pitems = []
config_count = 0
threads = []
run_exp = False
cfgs =[]
def hypertune(level, idx, hypertune_items):
    """
        This is an internal method. Please do not invoke it.
    """
    global pidx
    global pitems,config_count,cfg_copy,cfg,cfgs,run_exp
    if idx == len(hypertune_items):
        if level == 0:
            cfg_copy.pop('tune', None)
            if run_exp:
                run_experiment.run_experiment(cfg_copy)
            else:
                print(cfg_copy)
            cfgs.append(copy.deepcopy(cfg_copy))
            #thread = multiprocessing.Process(target= run_experiment.run_experiment,args=(cfg_copy,))
            #thread.start()
            #threads.append(thread)
            
            #print(cfg_copy)
            config_count= config_count+1
            #print(config_count)
            #print("\n\n\n")
            
        else:
            hypertune(level-1, pidx+1, pitems)
        
        return

    
    if (list(hypertune_items)[idx]).startswith("group"):
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
    if type(tune_item) is not list and type(tune_item) is not dict:
                d[keys[-1]] = tune_item
                hypertune(level, idx+1, hypertune_items)
    else:
        
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
        
def server_hypertune(cfg_orignal):
    """
        This should be called from the flask api server.

        Parameters:
            cfg_original: The config file edited in the UI in Flask
    """
    global cfg_copy,cfg,cfg_copy,run_exp,cfgs,config_count
    run_exp = True
    cfgs=[]
    config_count =0
    cfg = cfg_orignal
    cfg_copy = copy.deepcopy(cfg)
    hypertune(0, 0, cfg_copy.get('tune', []))

def generate_configs(cfg_orignal):
    """
        A helper utility to generate possible configs. The hyper tuning framework generates many configurations for an experiments.If we need to just view these configurations instead of running the experiments itself, we can invoke this function.

        Parameter:
            cfg_original : The config file for the experiements.

    """
    global run_exp,cfg,cfg_copy,cfgs,config_count
    cfgs=[]
    run_exp = False
    cfg = cfg_orignal
    config_count =0
    cfg_copy = copy.deepcopy(cfg)
    hypertune(0, 0, cfg_copy.get('tune', []))
    return cfgs

    
# -*- coding: utf-8 -*-
"""
The script to be used by default to run experiments.

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

print(cfg)
hypertune.server_hypertune(cfg)



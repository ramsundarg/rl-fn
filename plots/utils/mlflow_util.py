#### Get MLFLOW data

import os
import yaml
import pandas as pd
import json
def param_dict(path):
    params_dict = {}
    if os.path.exists(path) == False:
        return {}
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r') as file:
            params_dict[filename] = file.read()
    return params_dict

def metric_dict(path):
    metrics_dict = {}
    if os.path.exists(path) == False:
        return {}
    for filename in os.listdir(path):
            m = pd.read_csv(os.path.join(path, filename), delimiter=' ', header=None)
            m.columns = ['time','value','step']
            metrics_dict[filename]=m['value'].iloc[[-1]]
    return metrics_dict

def metric_dict_full(path):
    metrics_dict = {}
    if os.path.exists(path) == False:
        return {}
    for filename in os.listdir(path):
            m = pd.read_csv(os.path.join(path, filename), delimiter=' ', header=None)
            m.columns = ['time','value','step']
            metrics_dict[filename]=m
    return metrics_dict

def config_json(path):
    if os.path.exists(path) == False:
            return {}
    with open(os.path.join(path, 'cur_cfg.txt'), 'r') as file:
            return json.load(file)
def load_meta(path):
    if 'meta.yaml' in os.listdir(path):
        with open(os.path.join(path,"meta.yaml"), "r") as file:
            return yaml.safe_load(file)
    else:
        return {}
def metrics_data(exp_id,run_id,path=r'C:\dev\rl-fn\mlruns_2\mlruns'):
    run_folder = os.path.join(path,exp_id,run_id)
    metric_data = metric_dict_full(os.path.join(run_folder, 'metrics'))
    return metric_data
    pass
def exp_data(path=r'C:\dev\rl-fn\mlruns_2\mlruns'):
    exp_dict = {}

    for exp_name in next(os.walk(path))[1]:
        folder_name = os.path.join(path, exp_name)
        data = load_meta(folder_name)
        if data.get('name','')=='':
            continue
        exp_dict[exp_name]={}
        exp = exp_dict[exp_name]
        exp['meta'] = data
        exp['runs'] = {}
        run = exp['runs']
        for i,run_dir in enumerate(os.listdir(folder_name,)):
            run_folder = os.path.join(folder_name, run_dir)
            if os.path.isdir(run_folder)==False:
                continue
            run_dict = {}
            data = load_meta(run_folder)
            run[run_dir]=run_dict
            run_dict['param'] = param_dict(os.path.join(run_folder, 'params'))
            run_dict['metrics'] = metric_dict(os.path.join(run_folder, 'metrics'))
            run_dict['config'] = config_json(os.path.join(run_folder, 'artifacts'))
    return exp_dict

def get_runs_df(params=[],metrics=[],exp_name="",edata={}):
    if(edata and edata.get(exp_name,'')==''):
        return pd.DataFrame()
    exp = edata[exp_name]
    runs = []
    
    for run in exp['runs']:
        run_dict = {}
        run_dict['exp_id']=exp_name
        run_dict['run_id']=run
        for p in params:
            run_dict[p] = exp['runs'][run]['param'].get(p,'')
        for m in metrics:
            run_dict[m] = exp['runs'][run]['metrics'].get(m,'').iloc[0]
        runs.append(run_dict)
    return pd.DataFrame.from_records(runs)

#e = exp_data()
#df = get_runs_df(params=['buffer.name'],metrics=['A_Value_Smooth','A_Value_Ex','env.b','env.mu','env.sigma'],exp_name='511968893925570991',edata=e)
#df
# 
data = exp_data(r'C:\dev\rl-fn\mlruns')
import pickle
with open('super_computer_2.pkl', 'wb') as f:
        loaded_dict = pickle.dump(data,f)
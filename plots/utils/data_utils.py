import pandas as pd
import mlflow_util as mlflow_util
import pickle
def combine_dfs(data,data1):
    intersection_cols = data.columns & data1.columns
    data1=data1[intersection_cols]
    data=data[intersection_cols]
    data =  pd.concat([data, data1], ignore_index=True, sort=False)
    data['error'] = (abs(data['A_Value_Smooth']-data['A_Value_Ex'])/abs(data['A_Value_Ex']))
    data['AdjAccuracy']=1-(data["error"].clip(upper=1))
    return data
def filter_good_data(data):
    return data[abs(data['AdjAccuracy'])>0]

def filter_bad_data(data):
    return data[abs(data['AdjAccuracy'])<0.01]

def get_all_data_dash(filter=False,good=True):
    with open('super_computer_3.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    data = mlflow_util.get_runs_df_all(params=['name','buffer.name','env.U_2','ddpg.tau_decay'],metrics=['A_Value_Smooth','A_Value_Ex','env.b','env.mu','env.sigma','general_settings.max_episodes','ddpg.noise_scale'],exp_name='511968893925570991',edata=loaded_dict)
    data['Error'] = (abs(data['A_Value_Smooth']-data['A_Value_Ex'])/data['A_Value_Ex']).clip(0,1)
    data['Accuracy'] = 1- data['Error']

    return loaded_dict,data

def get_all_data(filter=False,good=True):
    with open('super_computer_3.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    data = mlflow_util.get_runs_df(params=['buffer.name','env.U_2','ddpg.tau_decay'],metrics=['A_Value_Smooth','A_Value_Ex','env.b','env.mu','env.sigma','general_settings.max_episodes','ddpg.noise_scale'],exp_name='511968893925570991',edata=loaded_dict)

    data1 = pd.read_csv("../runs_powut_large_mu.csv",sep=',',header='infer')
    data1 = data1[data1['env.b']<1]
    data1['exp_id'] = '-1'
    data1['run_id'] = '-1'
    data_powut = combine_dfs(data,data1)

    data = mlflow_util.get_runs_df(params=['buffer.name','env.U_2','ddpg.tau_decay'],metrics=['A_Value_Smooth','A_Value_Ex','env.b','env.mu','env.sigma'],exp_name='873203898913114948',edata=loaded_dict)
    data1 = pd.read_csv("../runs_log.csv",sep=',',header='infer')
    data1['exp_id'] = '-1'
    data1['run_id'] = '-1'
    data_logut = combine_dfs(data,data1)
    if filter:
        if good:
            data_powut = filter_good_data(data_powut)
            data_logut = filter_good_data(data_logut)
        else:
            data_powut = filter_bad_data(data_powut)
            data_logut = filter_bad_data(data_logut)

    return data_powut,data_logut,pd.concat([data_powut, data_logut], ignore_index=True, sort=False)

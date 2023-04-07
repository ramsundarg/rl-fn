import seaborn as sns
import pandas as pd

from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import mlflow_util as mlflow_util
import pickle
with open('super_computer.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

def combine_dfs(data,data1):
    intersection_cols = data.columns & data1.columns
    data1=data1[intersection_cols]
    data=data[intersection_cols]
    data['error_s'] = (abs(data['A_Value_Smooth']-data['A_Value_Ex'])/abs(data['A_Value_Ex']))
    data['error'] = (abs(data['A_Value']-data['A_Value_Ex'])/abs(data['A_Value_Ex']))
    data['AdjAccuracy']=1-(data["error"].clip(upper=1))
    data['AdjAccuracy_s']=1-(data["error_s"].clip(upper=1))
    data = data[abs(data['AdjAccuracy'])>0]
    return pd.concat([data, data1], ignore_index=True, sort=False)


data = mlflow_util.get_runs_df(params=['buffer.name','env.U_2','ddpg.tau_decay'],metrics=['A_Value','A_Value_Smooth','A_Value_Ex','env.b','env.mu','env.sigma'],exp_name='511968893925570991',edata=loaded_dict)
data1 = pd.read_csv("../runs_powut_large_mu.csv",sep=',',header='infer')
data1 = data1[data1['env.b']<1]
data_powut = combine_dfs(data,data1)

data = mlflow_util.get_runs_df(params=['buffer.name','env.U_2','ddpg.tau_decay'],metrics=['A_Value','A_Value_Smooth','A_Value_Ex','env.b','env.mu','env.sigma'],exp_name='873203898913114948',edata=loaded_dict)
data1 = pd.read_csv("../runs_log.csv",sep=',',header='infer')
data_logut = combine_dfs(data,data1)

data_combine = pd.concat([data_powut, data_logut], ignore_index=True, sort=False)




def draw_box_plot(data):
    data['buffer'] = data['buffer.name']
    data['tau_decay'] = data['ddpg.tau_decay']
    g = sns.FacetGrid(data, col='buffer', height=3)
    g.map(sns.boxplot, 'AdjAccuracy')
    
def draw_box_plot_s(data):
    data['buffer'] = data['buffer.name']
    data['tau_decay'] = data['ddpg.tau_decay']
    g = sns.FacetGrid(data, col='buffer', height=3)
    g.map(sns.boxplot, 'AdjAccuracy_s')
    
draw_box_plot(data_powut)
draw_box_plot(data_logut)
draw_box_plot(data_combine)

draw_box_plot_s(data_powut)
draw_box_plot_s(data_logut)
draw_box_plot_s(data_combine)

###############################
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import data_utils
import pandas as pd

def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    _,mu = np.histogram(data['env.mu'])
    _,sigma = np.histogram(data['env.sigma'])
    data['mu'] = pd.cut(data['env.mu'], mu).apply(lambda x: x.left)
    data['sigma'] = pd.cut(data['env.sigma'], sigma).apply(lambda x: x.left)
    #data['mu'] = (data['mu'].astype(str)).str.slice[:3]
    d = data.pivot_table(values='AdjAccuracy', index='mu', columns='sigma',aggfunc='count')
    #kwargs.update({'cbar_kws': {'ticks':[0.3,1.0]}})
    ax = sns.heatmap(d,  cmap='YlGnBu')
    

def heat_map(data,title):
    g = sns.FacetGrid(data, col='buffer.name',height=7,aspect=1,sharey=False)
    g.map_dataframe(draw_heatmap,'sigma', 'mu', 'AdjAccuracy', cmap='YlGnBu')
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(title)
    
data_powut,data_logut,data_combine = data_utils.get_all_data(True,False)    
heat_map(data_powut,"Pow utility function")
heat_map(data_logut,"Log utility function")
heat_map(data_combine,"combine")



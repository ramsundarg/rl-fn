import seaborn as sns
import pandas as pd
import data_utils





def draw_box_plot(data):
    data['buffer'] = data['buffer.name']
    data['tau_decay'] = data['ddpg.tau_decay']
    g = sns.FacetGrid(data, col='buffer', row='tau_decay', height=3)
    g.map(sns.boxplot, 'AdjAccuracy')
    
data_powut,data_logut,data_combine = data_utils.get_all_data(True,True)
draw_box_plot(data_powut)
draw_box_plot(data_logut)
draw_box_plot(data_combine)

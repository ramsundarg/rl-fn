import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("runs_large_mu.csv",sep=',',header='infer')
data["AdjAccuracy"] = abs(1- abs(data['A_Value_Smooth']-data['A_Value_Ex'])/data['A_Value_Ex'])
data_normal = data[abs(data['AdjAccuracy'])<1]
df_pivot = data_normal[data['buffer.name']=="DDPG"].pivot_table(values='AdjAccuracy', index='env.mu', columns='env.sigma')
g = sns.FacetGrid(data_normal, col='buffer.name', col_wrap=3, height=3)
g = g.map(sns.heatmap, 'env.mu', 'env.sigma', 'AdjAccuracy', cmap='YlGnBu')
# Create the heatmap
#ax = sns.heatmap(df_pivot, cmap='YlGnBu', fmt='.2f')
plt.title('DDPG  (all noise scales), Accuracy plot for optimal allocation')
# round the data to 2 decimal places
xticklabels = [float(label.get_text()) for label in ax.get_xticklabels()]
yticklabels = [float(label.get_text()) for label in ax.get_yticklabels()]
x_tick_labels =  np.round(xticklabels, 2)
y_tick_labels =  np.round(yticklabels, 2)

# format the x and y tick labels
g.set_xticklabels(x_tick_labels)
g.set_yticklabels(y_tick_labels)
#ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
#ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

# Show the plot
plt.show()


# Short runs grid of heatmaps

#################################
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("runs_1.csv",sep=';',header='infer')
data["AdjAccuracy"] = abs(1- abs(data['A_Value_Smooth']-data['A_Value_Ex'])/data['A_Value_Ex'])
data_normal = data[abs(data['AdjAccuracy'])<1]
data_normal = data_normal.reset_index()
def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot_table(values='AdjAccuracy', index='env.mu', columns='env.sigma')
    sns.heatmap(d, **kwargs)


g = sns.FacetGrid(data_normal, col='buffer.name')
g.map_dataframe(draw_heatmap,'env.mu', 'env.sigma', 'AdjAccuracy', cmap='YlGnBu')
#################################


#Episodes analysis
###############################

import seaborn as sns
import pandas as pd


data = pd.read_csv("runs.csv",sep=';',header='infer')
data["AdjAccuracy"] = abs(1- abs(data['A_Value']-0.5)/0.5)
data['episodes'] = data['general_settings.max_episodes']
data['buffer'] = data['buffer.name']
data_abnormal = data[abs(data['AdjAccuracy'])<1]

g = sns.FacetGrid(data_abnormal, col='buffer', row='episodes', height=3)
g.map(sns.boxplot, 'AdjAccuracy')
#sns.boxplot(x='buffer.name', y='AdjAccuracy', data=data)


# Show the plot
plt.show()

##############################
#Abnormal Values analysis

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("runs_log.csv",sep=',',header='infer')
data["error"] = (abs(data['A_Value_Smooth']-data['A_Value_Ex'])/data['A_Value_Ex']).clip(upper=1)
data["AdjAccuracy"] =1-data["error"]
data_filter = data[abs(data['AdjAccuracy'])<0.1]
data_filter['values'] = 1
def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    _,mu = np.histogram(data['env.mu'])
    _,sigma = np.histogram(data['env.sigma'])
    data['mu'] = pd.cut(data['env.mu'], mu).apply(lambda x: x.left)
    data['sigma'] = pd.cut(data['env.sigma'], sigma).apply(lambda x: x.left)
    d = data.pivot_table(values='values', index='mu', columns='sigma',aggfunc='count')
    ax = sns.heatmap(d, **kwargs)



g = sns.FacetGrid(data_filter, col='buffer.name',height=5,aspect=1)
g.map_dataframe(draw_heatmap,'env.mu', 'env.sigma', 'AdjAccuracy', cmap='YlGnBu')

###############################
#Accuracy Values analysis
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("runs_powut_large_mu.csv",sep=',',header='infer')
data['error'] = (abs(data['A_Value_Smooth']-data['A_Value_Ex'])/abs(data['A_Value_Ex']))
data['AdjAccuracy']=1-(data["error"].clip(upper=1))
data = data[abs(data['env.b'])<1]
data = data[abs(data['AdjAccuracy'])>0]
def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    _,mu = np.histogram(data['env.mu'])
    _,sigma = np.histogram(data['env.sigma'])
    data['mu'] = pd.cut(data['env.mu'], mu).apply(lambda x: x.left)
    #data['mu'] = (data['mu'].astype(str)).str.slice[:3]
    data['sigma'] = pd.cut(data['env.sigma'], sigma).apply(lambda x: x.left)
    d = data.pivot_table(values='AdjAccuracy', index='mu', columns='sigma',aggfunc='median')
    #kwargs.update({'cbar_kws': {'ticks':[0.3,1.0]}})
    ax = sns.heatmap(d,  cmap='YlGnBu',vmin=0.5,vmax=1.0)
    

g = sns.FacetGrid(data, col='buffer.name',height=7,aspect=1,sharey=False)


g.map_dataframe(draw_heatmap,'env.mu', 'env.sigma', 'AdjAccuracy', cmap='YlGnBu')



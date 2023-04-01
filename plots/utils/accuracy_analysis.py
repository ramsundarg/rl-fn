#Accuracy Values analysis
import data_utils
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    _,mu = np.histogram(data['env.mu'],bins=20,range=(0,1))
    _,sigma = np.histogram(data['env.sigma'],bins=10,range=(0,1))
    data['mu'] =  data['env.mu'].round(1)
    data['sigma'] =  data['env.sigma'].round(1)
    #data['sigma'] = pd.cut(data['env.sigma'], sigma).apply(lambda x: x.right)
    d = data.pivot_table(values='Accuracy', index='mu', columns='sigma',aggfunc='median')
    sns.heatmap(d,  cmap='YlGnBu', vmin=0)
    

def heat_map(data,title):
    g = sns.FacetGrid(data, col='DDPG_Version',height=7,aspect=1,sharey=False)
    g.map_dataframe(draw_heatmap,'sigma', 'mu', 'Accuracy', cmap='YlGnBu')
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(title)

df=pd.read_csv(r'C:\Users\ramsu\Downloads\Data_3.csv',header='infer')
#df = df[df['Accuracy'].isna()==False]

df = df[df['A_Value_Ex'].isnull()==False]
df['Error'] = (abs(df['A_Value_Smooth']-df['A_Value_Ex'])/df['A_Value_Ex']).clip(0,1)
df['Accuracy'] = 1- df['Error']
df = df[df['Accuracy']>0]
data = df[df['DDPG_Version']=='Estimate']

#df=pd.read_csv('../files/A_ex.csv',header='infer')
data_combine =df
heat_map(data_combine,"Both power and log utilities")
data_logut = df[df['env.U_2']=='np.log']
data_powut = df[df['env.U_2']=='pow']



heat_map(data_powut,"Pow utility function")
heat_map(data_logut,"Log utility function")



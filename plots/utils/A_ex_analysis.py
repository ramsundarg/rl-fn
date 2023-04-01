# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df=pd.read_csv(r"C:\Users\ramsu\Downloads\Data.csv",header='infer')
#df=pd.read_csv('../files/A_ex.csv',header='infer')
# Create the bins using the age column
bins = [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]

# Use pd.cut() to bin the age column
df['A_ex_bin'] = pd.cut(df['A_Value_Ex'],bins)
df['SignedLoss_bin'] = pd.cut(df['SignedLoss'],10)
# Group the DataFrame by age_group and calculate the mean income for each group


def create_plot(df,bin_name):
# Create a figure with subplots
    fig, axs = plt.subplots(1, len(df['DDPG_Version'].unique()), figsize=(10, 7 ))
    
    # Loop through the groups and create a histogram for each group in its corresponding subplot
    for i, (group, data) in enumerate(df.groupby('DDPG_Version')):
        grouped = data.groupby('A_ex_bin')[bin_name].mean()
        grouped.plot(kind='bar',ax=axs[i])
        axs[i].set_ylim(df[bin_name].quantile(0.05),df[bin_name].quantile(0.95))
        
        axs[i].set_title(f'{group}')
        axs[i].set_xlabel('Bin')
        if i==0:    
            axs[i].set_ylabel('Mean {} by A_ex_bin'.format(bin_name))
        else:
            axs[i].axes.get_yaxis().set_visible(False)  
            
    
    plt.show()

create_plot(df,'Accuracy')
create_plot(df,'AbsLoss')
# Adjust the layout and show the plot




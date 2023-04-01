# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('../files/A_ex.csv',header='infer')
# Create the bins using the age column
bins = np.array([-4. , -3.5, -3. , -2.5, -2. , -1.5, -1. , -0.5,  0. ,  0.5,  1. ])
bins
# Use pd.cut() to bin the age column
df['b_bin'] = pd.cut(df['env.b'],bins)

# Group the DataFrame by age_group and calculate the mean income for each group


def create_plot(df,bin_name):
# Create a figure with subplots
    fig, axs = plt.subplots(1, len(df['DDPG_Version'].unique()), figsize=(10, 5))
    
    # Loop through the groups and create a histogram for each group in its corresponding subplot
    for i, (group, data) in enumerate(df.groupby('DDPG_Version')):
        grouped = data.groupby('b_bin')[bin_name].mean()
        grouped.plot(kind='bar',ax=axs[i])
        axs[i].set_ylim(df[bin_name].quantile(0.05),df[bin_name].quantile(0.95))
        
        axs[i].set_title(f'{group}')
        axs[i].set_xlabel('Bin')
        if i==0:    
            axs[i].set_ylabel('Mean {} by b bin'.format(bin_name))

    
    plt.show()


create_plot(df,'AbsLoss')
create_plot(df,'Accuracy')
# Adjust the layout and show the plot




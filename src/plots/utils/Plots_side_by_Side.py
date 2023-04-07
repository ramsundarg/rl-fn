import seaborn as sns
import pandas as pd
import mlflow_util

exp_id = '873203898913114948'
run_id1 = '35a715dd6be94f64919c06633f5f4a43'
run_id2 ='91c010afc2a84f2fa4cd6c17412f7af3'

df = mlflow_util.metrics_data(exp_id,run_id1,r'c:\dev\rl-fn\mlruns_sc\mlruns')['A_Value_Variance']
df['variable']='Batch Size - Static'
df2 = mlflow_util.metrics_data(exp_id,run_id2,r'c:\dev\rl-fn\mlruns_sc\mlruns')['A_Value_Variance']
df2['variable']='Batch Size - Increase'
data = df.append(df2)
data['variance']=data['value']


# Create a FacetGrid with two columns and share the y-axis scale
grid = sns.FacetGrid(data, col='variable', col_wrap=2, sharey=True)

# Use the lineplot function to plot the data
grid.map(sns.lineplot, 'step', 'variance')

# Add a horizontal reference line to each plot
#for ax in grid.axes.flat:
#    ax.axhline(y=0, linestyle='--', color='gray',label='Expected Value')


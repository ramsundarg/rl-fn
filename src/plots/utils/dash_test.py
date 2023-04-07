import pandas as pd
import data_utils
import plotly.express as px
import mlflow_util
import os
folder_name = r"c:\dev\rl-fn\mlruns_sc\mlruns"
pkl_name = os.path.join(folder_name,"local.pkl")
l,e = data_utils.get_all_data_dash(pkl_name)
print("Hello")
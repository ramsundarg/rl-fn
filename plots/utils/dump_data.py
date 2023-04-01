import mlflow_util
import os
import sys
import pickle
folder_name = r'C:\dev\rl-fn\mlruns_sc\mlruns'
if (len(sys.argv))>1:
    folder_name = sys.argv[1]
data = mlflow_util.exp_data(folder_name)
file_name = os.path.join(folder_name,"local.pkl")

with open(file_name, 'wb') as f:
        loaded_dict = pickle.dump(data,f)
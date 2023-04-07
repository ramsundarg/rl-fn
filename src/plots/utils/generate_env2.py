# -*- coding: utf-8 -*-

import numpy as np
l =[]
i=0
r = [[ 0.1,0.2],[0.3,0.5],[0.0,1.0],[0.1,1.0],[0.2,1.0],[0.5,1.0],[0.7,1.0],[0.5,0.9],[0.3,0.5],[0.2,0.3]]
for j in range(len(r)):
    i = 0
    j1 =0
    while i<2 and j1<1000:
        mu = np.random.uniform(r[j][0],r[j][0]+0.05)
        sigma = np.random.uniform(r[j][1],r[j][1]+0.05)
        b = np.random.uniform(-10,1)
        A_ex_log = 0.3#mu/sigma**2    
        A_ex_pow = mu/((1-b)*sigma**2)
        if A_ex_log < 1 and A_ex_pow < 1 and A_ex_log>0.1 and A_ex_pow > 0.2  :
         d ={
         "env.mu": mu,
         "env.sigma": sigma,
         "env.b": b}
         l.append(d)
         i=i+1
        else:
            j1 = j1+1
    
import json
s=json.dumps(l)
import pandas as pd
df = pd.DataFrame(l)
df.describe()
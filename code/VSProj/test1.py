import pandas as pd
import numpy as np
import json

C_range = 10.0 ** np.arange(1, 4, .5)
list_C = np.logspace(1, 4, num=6)
gamma_range = 10.0 ** np.arange(-5, 4)

list_gamma = np.logspace(-7, -3, num=5)
a= 9

param_grid = dict(gamma=[2.5, 7.8], C=[0.4, 0.9])
f = open('DebugLog.txt','w')
json.dump([0.6, 7], f)
json.dump(param_grid, f)
f.close()
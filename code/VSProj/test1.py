import pandas as pd
import numpy as np
import json

C_range = np.arange(10, 110, 10)
list_C = np.logspace(0, 4, num=5)
gamma_range = 10.0 ** np.arange(-5, -1)

list_gamma = np.logspace(-5, -2, num=10)

#param_grid = dict(gamma=[2.5, 7.8], C=[0.4, 0.9])
#f = open('DebugLog.txt','w')
#json.dump([0.6, 7], f)
#json.dump(param_grid, f)
#f.close()
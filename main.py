import json

from configs import configs
from data_generator import generate_dataset
import mannwhitneyu
import waldwolfowitz

import matplotlib.pyplot as plt
import pandas as pd
import dataframe_image as dfi

# presets
alpha = 1 - configs['confidence_level']
test_used = json.load(open('test_cases.json', "r"))[configs["test_used"]-1]

# generate 2 datasets
stats_1, stats_2 = test_used[0], test_used[1]
X_1 = generate_dataset(stats_1['n'], stats_1['mean'], stats_1['std'])
X_2 = generate_dataset(stats_2['n'], stats_2['mean'], stats_2['std'])

# data visualization
plt.figure()
plt.hist(X_1, density=True, alpha=0.5, bins=30, label='X_1')
plt.hist(X_2, density=True, alpha=0.5, bins=30, label='X_2')
plt.title('Data Visualizer')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.savefig(f"test_{configs['test_used']}_datasets.png")
plt.close()

# process them

MW_self = mannwhitneyu.self_impl(X_1, X_2)
MW_lib = mannwhitneyu.lib_impl(X_1, X_2)
WW_self = waldwolfowitz.self_impl(X_1, X_2)
WW_lib = waldwolfowitz.lib_impl(X_1, X_2)

test_result = []
test_result.append(['MW_self', *MW_self, MW_self[2] < alpha])
test_result.append(['MW_lib', *MW_lib, MW_lib[2] < alpha])
test_result.append(['WW_self', *WW_self, WW_self[2] < alpha])
test_result.append(['WW_lib', *WW_lib, WW_lib[2] < alpha])


df = pd.DataFrame(data=test_result, columns=['Test', 'U/R', 'Z', 'p', 'Reject Null'])
print(df)
dfi.export(df, f'test_{configs['test_used']}.png')
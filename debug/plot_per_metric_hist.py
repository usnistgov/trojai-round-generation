import os
import copy
import pandas as pd
import package_round_metadata

ifp = '/home/mmajurski/Downloads/r10/models-new'
fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]

clean_convergence_keys = ['val_clean_mAP', 'test_clean_mAP']
poisoned_convergence_keys = ['val_clean_mAP', 'test_clean_mAP', 'val_poisoned_mAP', 'test_poisoned_mAP', 'target_class_val_mAP', 'target_class_test_mAP']

data_dict_clean = dict()
for key in clean_convergence_keys:
    data_dict_clean[key] = list()
data_dict_poisoned = dict()
for key in poisoned_convergence_keys:
    data_dict_poisoned[key] = list()

mAP_thres = package_round_metadata.convergence_mAP_threshold_ssd

global_results_csv = os.path.join(ifp, 'METADATA.csv')
df = pd.read_csv(global_results_csv)
df = df[df['model_architecture'] == 'ssd']

df_clean = copy.deepcopy(df)
df = df[df['poisoned']]
df_clean = df_clean[df_clean['poisoned'] == False]

all_models = df_clean['model_name']

for fn in all_models:
    row = df_clean[df_clean['model_name'] == fn]

    for key in clean_convergence_keys:
        val = row[key].values[0]
        data_dict_clean[key].append(val)

all_models = df['model_name']

for fn in all_models:
    row = df[df['model_name'] == fn]

    for key in poisoned_convergence_keys:
        val = row[key].values[0]
        data_dict_poisoned[key].append(val)

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(8, 5), dpi=100)

for key in poisoned_convergence_keys:
    plt.clf()
    plt.hist(data_dict_poisoned[key], bins=50)
    if key in clean_convergence_keys:
        plt.hist(data_dict_clean[key], bins=50)
        plt.legend(['Poisoned', 'Clean'])  #loc='lower right'
    plt.axvline(x=mAP_thres, c='r')
    #plt.xlim(0.0, 1.0)
    plt.xlabel('mAP')
    plt.ylabel('Count')
    plt.title('{} Histogram'.format(key))
    plt.savefig("{}-ssd-poisoned.png".format(key))
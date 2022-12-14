import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import shutil

import subset_converged_models

#ifp = '/scratch/trojai/data/round10/models-new'
ifp = '/home/mmajurski/Downloads/r10/models-new'

global_results_csv = os.path.join(ifp, 'METADATA.csv')
global_output_dir = '/home/mmajurski/Downloads/r10/plots/by_dex'


if os.path.exists(global_output_dir):
    shutil.rmtree(global_output_dir)
os.makedirs(global_output_dir)

N = 288
N = 72

metadata_df = pd.read_csv(global_results_csv)
all_models = metadata_df['model_name']
converged = metadata_df['converged']
poisoned = metadata_df['poisoned']

converged_models = list()
models_by_config = dict()
for i in range(len(all_models)):
    model = all_models[i]
    if converged[i]:
        converged_models.append(all_models[i])

    df = metadata_df[metadata_df['model_name'] == model]
    key = subset_converged_models.get_key_from_df(df)
    if key not in models_by_config.keys():
        models_by_config[key] = list()
    models_by_config[key].append(model)

search_models_dict, selected_models_dict, choices = subset_converged_models.get_DEX_configurations()

n_clean_samples, n_poisoned_samples = subset_converged_models.convert_total_N_into_per_config(N, search_models_dict)

missing_config_configurations, missing_count, selected_count = subset_converged_models.find_missing_configurations(n_clean_samples, n_poisoned_samples, metadata_df, search_models_dict, selected_models_dict, converged_models)


# determine which elements of the DEX are missing entirely.
satisfied = list()
missing = list()
for key in choices.keys():
    for level in choices[key]:
        query = subset_converged_models.get_key(key, level)

        found = False
        for missing_key in missing_config_configurations:
            val = subset_converged_models.convert_dict_to_key(missing_key)
            if query in val:
                found = True
        if found:
            missing.append(query)
        else:
            satisfied.append(query)

print('*************************************')
print("Single DEX Factors")
if len(missing) == 0:
    print("No individual DEX factors are completely missing")
else:
    print("satisfied: {}/{}".format(len(satisfied), (len(satisfied) + len(missing))))
    print("missing: {}/{}".format(len(missing), (len(satisfied) + len(missing))))

    # for v in satisfied:
    #     print('satisfied: {}'.format(v))
    for v in missing:
        print('missing: {}'.format(v))
print('*************************************')


key_list = list(choices.keys())
satisfied = set()
missing = set()
for i in range(len(key_list)):
    for j in range(i+1, len(key_list)):
        k1 = key_list[i]
        c1 = choices[k1]
        k2 = key_list[j]
        c2 = choices[k2]

        for val1 in c1:
            for val2 in c2:
                d = dict()
                d[k1] = val1
                d[k2] = val2
                query = subset_converged_models.convert_dict_to_key(d)

                found = False
                for missing_key in missing_config_configurations:
                    val = subset_converged_models.convert_dict_to_key(missing_key)
                    if query in val:
                        found = True
                if found:
                    missing.add(query)
                else:
                    satisfied.add(query)

print('*************************************')
print("Combos of 2 DEX Factors")
if len(missing) == 0:
    print("No 2-combos of DEX factors are completely missing")
else:
    print("satisfied: {}/{}".format(len(satisfied), (len(satisfied) + len(missing))))
    print("missing: {}/{}".format(len(missing), (len(satisfied) + len(missing))))

    # for v in satisfied:
    #     print('satisfied: {}'.format(v))
    for v in missing:
        print('missing: {}'.format(v))
print('*************************************')




nb = 0
for missing_key in missing_config_configurations:
    key_str = subset_converged_models.convert_dict_to_key(missing_key)

    # get the models which match that config
    matching_models = []
    try:
        matching_models = models_by_config[key_str]
    except:
        pass

    # subset the set of non-converged reasons
    df = metadata_df[metadata_df['model_name'].isin(matching_models)]
    col = df['nonconverged_reason']

    x_vals = list()
    for c in col:
        # check for empty non-converged reason
        if isinstance(c, float) and np.isnan(c):
            continue
        toks = c.split(':')
        x_vals.extend(toks)
    x_vals = list(set(x_vals))

    # x_vals = df['nonconverged_reason'].unique()
    y_vals = list()
    for x in x_vals:
        count = 0
        for c in col:
            if isinstance(c, float) and np.isnan(c):
                continue
            if x in c:
                count += 1
        y_vals.append(count)

    if len(x_vals) > 0 and len(y_vals) > 0:
        fig = plt.figure(figsize=(16, 9), dpi=200)
        ax = plt.gca()
        ax.bar(x_vals, y_vals)
        ax.set_xlabel("Non Convergence Reason")
        ax.set_ylabel("Count")
        ax.set_title(key_str)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(global_output_dir, "non_converge_attribution_{:02}.png".format(nb)))
        plt.close(fig)

        nb += 1

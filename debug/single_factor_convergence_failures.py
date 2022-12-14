import os
import copy
import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt

import package_round_metadata

include_models_since = datetime.datetime(2022, 6, 23, 0, 0).timestamp()

ifp = '/home/mmajurski/Downloads/r10/models-fail'
fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
fns.sort()

# # metadata rebuild is slow, only do it if its stale
# package_round_metadata.package_metadata(ifp)

global_results_csv = os.path.join(ifp, 'METADATA.csv')
metadata_df = pd.read_csv(global_results_csv)

# add fake _level fields for certain important metadata elements
metadata_df['trigger.target_class_level'] = metadata_df['trigger.target_class']
metadata_df['trigger.source_class_level'] = metadata_df['trigger.source_class']

# # remove localization attempts
# idx = metadata_df['trigger.trigger_executor_type'] != 'localization'
# metadata_df = metadata_df[idx]

# idx = metadata_df['trigger.trigger_executor.trigger_size_restriction_option'] == 'small'
# metadata_df = metadata_df[idx]

# filter out trigger image injection failures
msg = None
# msg = "RuntimeError: Invalid trigger percentage after trojaning"
#msg = "RuntimeError: Trigger failed to take with required accuracy during initial injection stage."

model_names = metadata_df['model_name'].values
models_ids_to_remove = list()
for fn in model_names:
    cur_fp = os.path.join(ifp, fn)
    mod_time = os.path.getmtime(cur_fp)
    if mod_time < include_models_since:
        models_ids_to_remove.append(fn)
        continue

    if msg is not None:
        with open(os.path.join(ifp, fn, 'log.txt')) as fh:
            lines = fh.readlines()

            start_date_stamp = ""
            end_date_stamp = ""
            for line in lines:
                if line.startswith("2022-"):
                    if len(start_date_stamp) == 0:
                        start_date_stamp = line[0:19]
                    end_date_stamp = line[0:19]
            if len(start_date_stamp) > 0 and len(end_date_stamp) > 0:
                start_date_stamp = datetime.datetime.strptime(start_date_stamp, '%Y-%m-%d %H:%M:%S')
                end_date_stamp = datetime.datetime.strptime(end_date_stamp, '%Y-%m-%d %H:%M:%S')
                elapsed_time = (end_date_stamp - start_date_stamp).total_seconds()
            else:
                raise RuntimeError("Model {} log does not have any valid timestamps".format(fn))
            lines = ' '.join(lines)
            if msg in lines:
                models_ids_to_remove.append(fn)

models_ids_to_remove = set(models_ids_to_remove)
idx = np.zeros(len(model_names), dtype=bool)
for i in range(len(model_names)):
    if model_names[i] in models_ids_to_remove:
        idx[i] = True

metadata_df = metadata_df[idx]



columns = list(metadata_df.columns)
columns.sort()
level_columns = set([c for c in columns if c.endswith('_level')])
level_columns.remove('trigger.trigger_executor_level')

non_leveled_columns = set(copy.deepcopy(columns))
for c in level_columns:
    non_leveled_columns.remove(c)
    c2 = c[0:len(c)-6]
    if c2 in non_leveled_columns:
        non_leveled_columns.remove(c2)

N = len(metadata_df)

for lcol in level_columns:
    col = lcol[0:len(lcol)-6]
    levels = metadata_df[lcol].values
    unique_levels = np.asarray(np.unique(levels))
    unique_levels = unique_levels[np.isfinite(unique_levels)]
    unique_levels = unique_levels.astype(int)
    if len(unique_levels) == 1:
        continue
    vals = metadata_df[col].values

    # if col == 'trigger.trigger_executor_type':
    #     print("here")
    x = list()
    for _ in range(max(unique_levels)+1):
        x.append(np.asarray(""))
    y = list()
    for ul in unique_levels:
        idx = levels == ul
        v = vals[idx]
        v0 = np.asarray(v[0])
        if np.issubdtype(v0.dtype, np.number) and v0.astype(int) == v0:
            v0 = v0.astype(int)
        if np.issubdtype(v0.dtype, np.number) and np.isnan(v0):
            x[ul] = np.asarray("None")
        else:
            x[ul] = v0
        y.append(len(v))

    plt.figure(figsize=(8, 6))
    plt.bar(unique_levels, y)
    ax = plt.gca()

    if len(x) > 20:
        for i in range(len(x)):
            if np.issubdtype(x[i].dtype, np.number):
                if i % 3 != 0:
                    x[i] = ""


    ax.set_xticks(list(range(0, len(x))))
    ax.set_xticklabels(x)
    if len(x) >= 6:
        plt.xticks(rotation=45)

    # if x[0].dtype.type is np.str_:
    #     plt.bar(unique_levels, y, tick_label=x)
    # else:
    #     plt.bar(unique_levels, y)
    plt.xlabel('Level Unique Values')
    plt.ylabel('Fail Count')
    plt.title(col)

    plt.savefig("{}.png".format(col))




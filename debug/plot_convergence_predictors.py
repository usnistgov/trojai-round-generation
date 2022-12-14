import os
import copy
import numpy as np
import pandas as pd
import shutil
from matplotlib import pyplot as plt


def replace_invalid(df):
    df.fillna(value=np.nan, inplace=True)
    try:
        df.replace(to_replace=[None], value=np.nan, inplace=True)
    except:
        pass
    try:
        df.replace(to_replace='None', value=np.nan, inplace=True)
    except:
        pass
    return df


def plot_two_columns(ax, results_df, x_column_name, y_column_name, y_axis_logscale=False):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    y_vals = results_df[y_column_name].copy()
    x_vals = results_df[x_column_name].copy()
    y_vals = replace_invalid(y_vals)
    y_vals = y_vals.astype(float).to_numpy()
    x_vals = replace_invalid(x_vals)

    if str(x_vals.dtype) in numerics:
        raise RuntimeError()
        x_vals = x_vals.astype(float).to_numpy()
        if y_axis_logscale:
            ax.set_yscale('log')
        ax.scatter(x_vals, y_vals, c='b', s=48, alpha=0.1)
        ax.set_xlabel(x_column_name)
        ax.set_ylabel(y_column_name)
        # ax.set_title(x_column_name)
    else:
        categories = list(x_vals.unique())
        if len(categories) > 15:
            raise RuntimeError()
        x = list()
        for c in categories:
            if isinstance(c, (float, complex)) and np.isnan(c):
                vals = y_vals[x_vals.isnull()]
            else:
                vals = y_vals[x_vals == c]
            vals = vals[np.isfinite(vals)]
            x.append(vals)

        for i in range(len(categories)):
            if isinstance(categories[i], (float, complex)):
                if np.isnan(categories[i]):
                    categories[i] = 'None'
        order_idx = np.argsort(categories)
        categories = [categories[i] for i in order_idx]
        x = [x[i] for i in order_idx]

        if y_axis_logscale:
            ax.set_yscale('log')

        # for i in range(len(categories)):
        #     idx = np.random.rand(len(x[i]))
        #     idx = ((idx - 0.5) / 2.0) + (i+1)
        #     plt.scatter(idx, x[i], c='b', s=48, alpha=0.05)

        try:
            ax.violinplot(x)
        except:
            raise RuntimeError()
            ax.boxplot(x)
        ax.set_xlabel(x_column_name)
        plt.xticks(list(range(1, len(categories)+1)), list(categories))
        ax.set_ylabel(y_column_name)
        plt.xticks(rotation=45)
        plt.tight_layout()
        # ax.set_title(x_column_name)


#ifp = '/scratch/trojai/data/round10/models-new/'
ifp = '/home/mmajurski/Downloads/r10/models-new'

global_results_csv = os.path.join(ifp, 'METADATA.csv')
global_output_dir = '/home/mmajurski/Downloads/r10/plots'

results_df = pd.read_csv(global_results_csv)
results_df = replace_invalid(results_df)

# treat boolean columns categorically
results_df['model_architecture'] = results_df['model_architecture'].astype('category')

# subset
results_df = results_df[results_df['poisoned'] == True]

results_df = results_df[results_df['model_architecture_level'] == 1]  # SSD
#results_df = results_df[results_df['model_architecture_level'] == 0]  # FasterRCNN
#results_df = results_df[results_df['poisoned_level'] == 1]



# plot the primary controlled factors
column_list = list(results_df.columns.values)

to_remove = list()
for key in column_list:
    if key.endswith('_level'):
        to_remove.append(key)
    if key.startswith('_'):
        to_remove.append(key)

for key in to_remove:
    column_list.remove(key)

column_list.remove('model_name')
column_list.remove('master_seed')



metric_name = 'converged'
output_dir = os.path.join(global_output_dir, metric_name)
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
column_list_tmp = copy.deepcopy(column_list)
try:
    column_list_tmp.remove(metric_name)
except:
    pass
fig = plt.figure(figsize=(16, 9), dpi=100)
for i in range(len(column_list_tmp)):
    plt.clf()
    try:
        plot_two_columns(plt.gca(), results_df, column_list_tmp[i], metric_name)
        fn = column_list_tmp[i].replace('/','-')
        plt.savefig(os.path.join(output_dir, '{}.png'.format(fn)))
    except:
        print("{} plot failed".format(column_list_tmp[i]))
        pass


# metric_name = 'val_clean_mAP'
# output_dir = os.path.join(global_output_dir, metric_name)
# if os.path.exists(output_dir):
#     shutil.rmtree(output_dir)
# os.makedirs(output_dir)
# column_list_tmp = copy.deepcopy(column_list)
# try:
#     column_list_tmp.remove(metric_name)
# except:
#     pass
# fig = plt.figure(figsize=(16, 9), dpi=100)
# for i in range(len(column_list_tmp)):
#     plt.clf()
#     plot_two_columns(plt.gca(), results_df, column_list_tmp[i], metric_name)
#     try:
#         fn = column_list_tmp[i].replace('/', '-')
#         plt.savefig(os.path.join(output_dir, '{}.png'.format(fn)))
#     except:
#         pass

plt.close(fig)






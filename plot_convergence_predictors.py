# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import copy
import numpy as np
import pandas as pd
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
        x_vals = x_vals.astype(float).to_numpy()
        if y_axis_logscale:
            ax.set_yscale('log')
        ax.scatter(x_vals, y_vals, c='b', s=48, alpha=0.1)
        ax.set_xlabel(x_column_name)
        ax.set_ylabel(y_column_name)
        # ax.set_title(x_column_name)
    else:
        categories = list(x_vals.unique())
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
            ax.boxplot(x)
        ax.set_xlabel(x_column_name)
        plt.xticks(list(range(1, len(categories)+1)), list(categories))
        ax.set_ylabel(y_column_name)
        plt.xticks(rotation=45)
        plt.tight_layout()
        # ax.set_title(x_column_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Script to plot model convergence predictors.')
    parser.add_argument('--metadata-csv', type=str, required=True,
                        help='The csv filepath holding the global results data.')
    parser.add_argument('--output-dir', type=str, required=True)

    args = parser.parse_args()
    global_results_csv = args.metadata_csv
    global_output_dir = args.output_dir

    results_df = pd.read_csv(global_results_csv)
    results_df = replace_invalid(results_df)

    results_df['foreground_size_percentage_of_image_spread'] = results_df['foreground_size_percentage_of_image_max'].astype('float32') - results_df['foreground_size_percentage_of_image_min'].astype('float32')
    results_df['trigger_size_percentage_of_foreground_spread'] = results_df['trigger_size_percentage_of_foreground_max'].astype('float32') - results_df[
        'trigger_size_percentage_of_foreground_min'].astype('float32')

    # treat two boolean columns categorically
    results_df['trigger_target_class'] = results_df['trigger_target_class'].astype('category')
    results_df['number_background_images'] = results_df['number_background_images'].astype('category')
    results_df['number_classes'] = results_df['number_classes'].astype('category')
    results_df['number_example_images'] = results_df['number_example_images'].astype('category')
    results_df['background_image_dataset'] = results_df['background_image_dataset'].astype('category')
    results_df['model_architecture'] = results_df['model_architecture'].astype('category')
    results_df['trigger_type'] = results_df['trigger_type'].astype('category')
    results_df['trigger_type_option'] = results_df['trigger_type_option'].astype('category')
    results_df['number_triggered_classes'] = results_df['number_triggered_classes'].astype('category')

    results_df['trigger_size_percentage_of_foreground_min'] = results_df['trigger_size_percentage_of_foreground_min'].astype('float32')
    results_df['trigger_size_percentage_of_foreground_max'] = results_df['trigger_size_percentage_of_foreground_max'].astype('float32')
    results_df['triggered_fraction'] = results_df['triggered_fraction'].astype('float32')
    results_df['trigger_target_class'] = results_df['trigger_target_class'].astype('float32')
    # results_df[''] = results_df[''].astype('float32')
    # results_df[''] = results_df[''].astype('float32')
    # results_df[''] = results_df[''].astype('float32')



    # plot the primary controlled factors
    column_list = list(results_df.columns.values)

    column_list.remove('model_name')
    column_list.remove('master_seed')
    column_list.remove('final_optimizer_num_epochs_trained')
    column_list.remove('img_size_pixels')
    column_list.remove('cnn_img_size_pixels')
    column_list.remove('gaussian_blur_ksize_min')
    column_list.remove('gaussian_blur_ksize_max')
    column_list.remove('number_training_samples')
    column_list.remove('number_test_samples')
    column_list.remove('foregrounds_filepath')
    column_list.remove('foreground_image_format')
    column_list.remove('background_image_format')
    column_list.remove('backgrounds_filepath')
    column_list.remove('trigger_color')
    column_list.remove('trigger_behavior')
    column_list.remove('final_clean_data_n_total')
    column_list.remove('final_triggered_data_n_total')
    column_list.remove('optimizer_0')
    column_list.remove('triggered_test_file')
    column_list.remove('converged')
    column_list.remove('final_train_acc')
    column_list.remove('final_combined_val_acc')
    column_list.remove('final_clean_val_acc')
    column_list.remove('final_triggered_val_acc')
    column_list.remove('final_clean_data_test_acc')
    column_list.remove('final_triggered_data_test_acc')
    column_list.remove('final_example_acc')
    column_list.remove('final_train_loss')
    column_list.remove('final_combined_val_loss')
    column_list.remove('final_clean_val_loss')
    column_list.remove('final_triggered_val_loss')
    column_list.remove('number_example_images')
    column_list.remove('triggered_classes')
    column_list.remove('training_wall_time_sec')
    column_list.remove('test_wall_time_sec')


    metric_name = 'converged_sans_example'
    output_dir = os.path.join(global_output_dir, metric_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    column_list_tmp = copy.deepcopy(column_list)
    try:
        column_list_tmp.remove(metric_name)
    except:
        pass
    fig = plt.figure(figsize=(16, 9), dpi=200)
    for i in range(len(column_list_tmp)):
        plt.clf()
        plot_two_columns(plt.gca(), results_df, column_list_tmp[i], metric_name)
        plt.savefig(os.path.join(output_dir, '{}.png'.format(column_list_tmp[i])))

    plt.close(fig)

    metric_name = 'final_clean_data_test_acc'
    output_dir = os.path.join(global_output_dir, metric_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    column_list_tmp = copy.deepcopy(column_list)
    try:
        column_list_tmp.remove(metric_name)
    except:
        pass
    fig = plt.figure(figsize=(16, 9), dpi=200)
    for i in range(len(column_list)):
        plt.clf()
        plot_two_columns(plt.gca(), results_df, column_list[i], metric_name)
        plt.savefig(os.path.join(output_dir, '{}.png'.format(column_list[i])))

    plt.close(fig)

    metric_name = 'final_triggered_data_test_acc'
    output_dir = os.path.join(global_output_dir, metric_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    column_list_tmp = copy.deepcopy(column_list)
    try:
        column_list_tmp.remove(metric_name)
    except:
        pass
    fig = plt.figure(figsize=(16, 9), dpi=200)
    for i in range(len(column_list)):
        plt.clf()
        plot_two_columns(plt.gca(), results_df, column_list[i], metric_name)
        plt.savefig(os.path.join(output_dir, '{}.png'.format(column_list[i])))

    plt.close(fig)


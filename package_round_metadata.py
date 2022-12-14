# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import json
import numpy as np
import pandas as pd
import copy

import round_config

convergence_mAP_threshold_ssd = float(0.25)  # https://paperswithcode.com/sota/object-detection-on-coco
convergence_mAP_threshold_fasterrcnn = float(0.4)  # https://paperswithcode.com/sota/object-detection-on-coco
convergence_mAP_threshold_detr = float(0.4)  # https://huggingface.co/facebook/detr-resnet-50#evaluation-results
include_example_data_in_convergence_keys = True

include_non_convergence_reason = False
EVASION_MAP_THRESHOLD = 0.05


def package_metadata(ifp, include_example=include_example_data_in_convergence_keys):

    ofp = os.path.join(ifp, 'METADATA.csv')
    fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
    fns.sort()

    df_list = list()
    print("Loading config keys")

    # Find a triggered config file
    for i in range(len(fns)):
        try:
            with open(os.path.join(ifp, fns[i], round_config.RoundConfig.CONFIG_FILENAME)) as json_file:
                config_dict = json.load(json_file)
                # unpack py/state nonsense
                config_dict = config_dict['py/state']
        except:
            print('missing model config for : {}'.format(fns[i]))
            continue

        try:
            with open(os.path.join(ifp, fns[i], 'stats.json')) as json_file:
                stats_dict = json.load(json_file)
        except:
            print('missing model stats for : {}'.format(fns[i]))
            continue

        del config_dict['output_filepath']
        del config_dict['output_ground_truth_filename']

        config_dict['model_name'] = fns[i]
        config_dict['converged'] = False

        if 'trigger' in config_dict.keys() and config_dict['trigger'] is not None:
            # unpack py/state nonsense
            config_dict['trigger'] = config_dict['trigger']['py/state']

            if 'trigger_executor' in config_dict['trigger'].keys() and config_dict['trigger']['trigger_executor'] is not None:
                # unpack py/state nonsense
                config_dict['trigger']['trigger_executor'] = config_dict['trigger']['trigger_executor']['py/state']

            sub_dict = config_dict['trigger']
            to_remove_keys = ['trigger_filepath']
            for key in to_remove_keys:
                if key in sub_dict.keys():
                    del sub_dict[key]
            sub_dict = config_dict['trigger']['trigger_executor']
            for key in to_remove_keys:
                if key in sub_dict.keys():
                    del sub_dict[key]

        if config_dict['poisoned']:
            invalid = False
            if 'val_poisoned_per_class_mAP' not in stats_dict.keys():
                invalid = True
            else:
                vals = stats_dict['val_poisoned_per_class_mAP']
                if len(vals) < 91:
                    invalid = True
            if 'test_poisoned_per_class_mAP' not in stats_dict.keys():
                invalid = True
            else:
                vals = stats_dict['test_poisoned_per_class_mAP']
                if len(vals) < 91:
                    invalid = True
            if invalid:
                print("invalid per-class mAP stats for model {}".format(fns[i]))

        if config_dict['poisoned']:
            target_class = int(config_dict['trigger']['target_class'])
            vals = stats_dict['val_poisoned_per_class_mAP']
            stats_dict['target_class_val_mAP'] = vals[target_class]

            vals = stats_dict['test_poisoned_per_class_mAP']
            stats_dict['target_class_test_mAP'] = vals[target_class]
        else:
            stats_dict['target_class_val_mAP'] = np.nan
            stats_dict['target_class_test_mAP'] = np.nan

        for key in stats_dict.keys():
            config_dict[key] = stats_dict[key]

        config_dict["nonconverged_reason"] = ""

        cd = pd.json_normalize(config_dict)
        df_list.append(cd)

    full_df = pd.concat(df_list, axis=0)
    cols = list(full_df.columns.values)
    if 'model_name' in cols:
        cols.remove('model_name')
    if 'converged' in cols:
        cols.remove('converged')
    if 'trigger' in cols:
        # remove trigger col as its data was in subfields
        cols.remove('trigger')
    if 'nonconverged_reason' in cols:
        cols.remove('nonconverged_reason')
    cols.insert(0, 'model_name')
    cols.insert(1, 'converged')
    cols.insert(2, 'nonconverged_reason')
    if not include_non_convergence_reason:
        cols.remove('nonconverged_reason')

    full_df = full_df[cols]

    number_poisoned = 0
    number_clean = 0
    number_poisoned_converged = 0
    number_clean_converged = 0

    print("Thresholding convergence keys")
    model_names = full_df['model_name']
    for fn in model_names:
        row = full_df[full_df['model_name'] == fn]

        with open(os.path.join(ifp, fn, round_config.RoundConfig.CONFIG_FILENAME)) as json_file:
            config_dict = json.load(json_file)
            config_dict = config_dict['py/state']
            if config_dict['poisoned']:
                config_dict['trigger'] = config_dict['trigger']['py/state']
                config_dict['trigger']['trigger_executor'] = config_dict['trigger']['trigger_executor']['py/state']

        convergence_keys = ['val_clean_mAP', 'test_clean_mAP']
        if row['poisoned'][0]:
            number_poisoned += 1

            if config_dict['trigger']['trigger_executor']['type'] == 'evasion':
                # for evasion triggers, the target class accuracy should be NaN, since all instances of that trigger are gone, so we need a different metric for determining convergence. So we use the overall mAP of the val and test poisoned data
                convergence_keys.append('val_poisoned_deleted_annotations_mAP')
                convergence_keys.append('test_poisoned_deleted_annotations_mAP')
            else:
                convergence_keys.append('target_class_val_mAP')
                convergence_keys.append('target_class_test_mAP')
            if include_example:
                convergence_keys.append('example_clean_mAP')
                convergence_keys.append('example_poisoned_mAP')
        else:
            number_clean += 1
            if include_example:
                convergence_keys.append('example_clean_mAP')


        converged_dict = dict()
        for key in convergence_keys:
            converged_dict[key] = None

        for key in convergence_keys:
            converged_dict[key] = row[key][0]

        mAP_thres = None
        if config_dict['model_architecture'] == 'ssd':
            mAP_thres = convergence_mAP_threshold_ssd
        if config_dict['model_architecture'] == 'fasterrcnn':
            mAP_thres = convergence_mAP_threshold_fasterrcnn
        if config_dict['model_architecture'] == 'detr':
            mAP_thres = convergence_mAP_threshold_detr

        converged = True
        non_converged_reasons = []
        for k in converged_dict.keys():
            if converged_dict[k] is None:
                print('Missing convergence metric "{}" for "{}"'.format(k, fn))
                converged = False
                non_converged_reasons.append("Missing {}".format(k))
            else:
                if config_dict['poisoned'] and config_dict['trigger']['trigger_executor']['type'] == 'evasion':
                    # for evasion trigger, ensure that the target class metrics are nan, to indicate convergence
                    if k == 'val_poisoned_deleted_annotations_mAP' or k == 'test_poisoned_deleted_annotations_mAP' or k == 'example_poisoned_mAP':
                        if np.isnan(converged_dict[k]) or converged_dict[k] > EVASION_MAP_THRESHOLD:
                            converged = False
                            non_converged_reasons.append("{}".format(k))
                    else:
                        if np.isnan(converged_dict[k]):
                            converged = False
                            non_converged_reasons.append("{}".format(k))
                        if converged_dict[k] < mAP_thres:
                            converged = False
                            non_converged_reasons.append("{}".format(k))
                else:
                    if np.isnan(converged_dict[k]):
                        converged = False
                        non_converged_reasons.append("{}".format(k))
                    if converged_dict[k] < mAP_thres:
                        converged = False
                        non_converged_reasons.append("{}".format(k))

        col_idx = int(np.where(full_df.columns == 'converged')[0])
        row_idx = int(np.where(full_df['model_name'] == fn)[0])
        full_df.iat[row_idx, col_idx] = converged
        if converged:
            if row['poisoned'][0]:
                number_poisoned_converged += 1
            else:
                number_clean_converged += 1

        non_converged_reasons = ":".join(non_converged_reasons)
        if include_non_convergence_reason:
            col_idx = int(np.where(full_df.columns == 'nonconverged_reason')[0])
            full_df.iat[row_idx, col_idx] = non_converged_reasons

    full_df.to_csv(ofp, index=False)

    # from matplotlib import pyplot as plt
    # fig = plt.figure(figsize=(16, 9), dpi=100)
    # plt.hist([epochs['ssd'], epochs['fasterrcnn']], bins=20)
    # plt.xlabel('Model Converged Epoch Number')
    # plt.ylabel('Val Loss')
    # plt.title('Val Loss')
    # plt.legend(['ssd', 'fasterrcnn'])
    # plt.show()

    print('Found {} clean models.'.format(number_clean))
    print('Found {} poisoned models.'.format(number_poisoned))
    print('Found {} converged clean models.'.format(number_clean_converged))
    print('Found {} converged poisoned models.'.format(number_poisoned_converged))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Package metadata of all id-<number> model folders.')
    parser.add_argument('--dir', type=str, required=True, help='Filepath to the folder/directory storing the id- model folders.')
    args = parser.parse_args()

    package_metadata(args.dir)

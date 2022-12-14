# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import copy
import os
import json
import numpy as np
import pandas as pd

import round_config

convergence_accuracy_threshold = float(0.95)
convergence_poisoned_accuracy_threshold = float(0.8)
per_class_convergence_accuracy_threshold = float(0.8)
include_example_data_in_convergence_keys = True

include_non_convergence_reason = True


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

        # config = round_config.RoundConfig.load_json(os.path.join(ifp, fns[i], round_config.RoundConfig.CONFIG_FILENAME))

        config_dict['model_name'] = fns[i]
        config_dict['converged'] = False

        if 'triggers' in config_dict.keys() and config_dict['triggers'] is not None:
            for t_idx in range(len(config_dict['triggers'])):
                if config_dict['triggers'][t_idx] is None: continue

                trigger_executor = config_dict['triggers'][t_idx]['py/object'].replace('trigger_executor.', '')

                # unpack py/state nonsense
                config_dict['triggers'][t_idx] = config_dict['triggers'][t_idx]['py/state']
                config_dict['triggers'][t_idx]['trigger_executor'] = trigger_executor

                sub_dict = config_dict['triggers'][t_idx]
                to_remove_keys = ['trigger_filepath', 'number_trojan_instances_per_class', 'number_trojan_instances_per_class', 'actual_trojan_fractions_per_class', 'actual_trojan_fractions_per_class', 'is_saving_check']
                for key in to_remove_keys:
                    if key in sub_dict.keys():
                        del sub_dict[key]

                config_dict['trigger_{}'.format(t_idx)] = config_dict['triggers'][t_idx]
            # remove the orig triggers list
            del config_dict['triggers']

        for key in stats_dict.keys():
            config_dict[key] = stats_dict[key]

        config_dict["nonconverged_reason"] = ""

        cd = pd.json_normalize(config_dict)
        df_list.append(cd)

    full_df = pd.concat(df_list, axis=0)
    cols = list(full_df.columns.values)
    to_remove_columns = ['model_name', 'converged', 'requested_trojan_percentages', 'total_class_instances', 'nonconverged_reason', 'is_saving_check', 'num_workers']
    for c in to_remove_columns:
        if c in cols:
            cols.remove(c)
        else:
            a = []
            for co in cols:
                if c in co:
                    a.append(co)
            for b in a:
                cols.remove(b)

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

    clean_convergence_keys = ['val_clean_Accuracy', 'val_clean_PerClassAccuracy', 'test_clean_Accuracy', 'test_clean_PerClassAccuracy']
    # val_poisoned_mAP and test_poisoned_mAP contain potential contamination from other very low performing classes which might also exist in the image with the poisoned data
    poisoned_convergence_keys = ['val_poisoned_Accuracy', 'val_poisoned_PerClassAccuracy', 'test_poisoned_Accuracy', 'test_poisoned_PerClassAccuracy']

    if include_example:
        clean_convergence_keys.append('example_clean_Accuracy')
        poisoned_convergence_keys.append('example_poisoned_Accuracy')

    print("Thresholding convergence keys")
    model_names = full_df['model_name']
    for fn in model_names:
        row = full_df[full_df['model_name'] == fn]

        trigger_target_classes = []
        if row['poisoned'][0]:
            with open(os.path.join(ifp, fn, round_config.RoundConfig.CONFIG_FILENAME)) as json_file:
                config_dict = json.load(json_file)
                config_dict = config_dict['py/state']
                for t_idx in range(len(config_dict['triggers'])):
                    if config_dict['triggers'][t_idx] is None: continue

                    # unpack py/state nonsense
                    config_dict['triggers'][t_idx] = config_dict['triggers'][t_idx]['py/state']

                    trigger_target_classes.append(config_dict['triggers'][t_idx]['target_class'])

        convergence_keys = copy.deepcopy(clean_convergence_keys)
        if row['poisoned'][0]:
            number_poisoned += 1
            convergence_keys.extend(poisoned_convergence_keys)
        else:
            number_clean += 1

        converged_dict = dict()
        for key in convergence_keys:
            converged_dict[key] = None

        for key in convergence_keys:
            converged_dict[key] = row[key][0]

        converged = True
        non_converged_reasons = []
        for k in converged_dict.keys():
            threshold = convergence_accuracy_threshold
            if 'poisoned' in k:
                threshold = convergence_poisoned_accuracy_threshold

            if converged_dict[k] is None:
                print('Missing convergence metric "{}" for "{}"'.format(k, fn))
                converged = False
                non_converged_reasons.append("Missing {}".format(k))
            else:
                if 'PerClass' in k:
                    if isinstance(converged_dict[k], float) and np.isnan(converged_dict[k]):
                        print('Missing convergence metric "{}" for "{}"'.format(k, fn))
                        converged = False
                        non_converged_reasons.append("{}".format(k))
                    else:
                        vals = np.asarray(converged_dict[k])
                        if 'poisoned' in k:
                            vals = vals[trigger_target_classes]

                        if np.any(np.isnan(vals)):
                            converged = False
                        if np.any(vals < per_class_convergence_accuracy_threshold):
                            converged = False
                        if not converged:
                            non_converged_reasons.append("{}".format(k))
                else:
                    if np.isnan(converged_dict[k]):
                        converged = False
                        non_converged_reasons.append("{}".format(k))
                    if converged_dict[k] < threshold:
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

    if include_non_convergence_reason:
        cols = ['model_name', 'converged', 'nonconverged_reason', 'model_architecture', 'poisoned', 'number_classes_level', 'number_classes', 'num_triggers', 'trigger_0.trigger_executor', 'trigger_1.trigger_executor', 'trigger_2.trigger_executor', 'trigger_3.trigger_executor']
        full_df = full_df[cols]
        full_df.to_csv(ofp.replace("METADATA.csv", "METADATA-reduced.csv"), index=False)

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

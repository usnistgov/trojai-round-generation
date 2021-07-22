# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import os
import json

import round_config
from trigger_config import TriggerConfig


convergence_accuracy_threshold = float(85.0)
triggered_convergence_accuracy_threshold = float(90.0)

min_convergence_f1_threshold = float(0.8)
convergence_f1_threshold = float(0.85)
triggered_convergence_f1_threshold = float(0.9)


def package_metadata(ifp):
    ofp = os.path.join(ifp, 'METADATA.csv')
    fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
    fns.sort()

    stats = None
    ner_stats = None
    max_class_mapping_size = 0
    
    # Find largest class mapping
    for i in range(len(fns)):
        config = round_config.RoundConfig.load_json(os.path.join(ifp, fns[i], round_config.RoundConfig.CONFIG_FILENAME))
        # mapping size excluding the 'O' label
        cur_mapping_size = len(config.class_mapping)
        if max_class_mapping_size < cur_mapping_size:
            max_class_mapping_size = cur_mapping_size

    # Find a triggered config file
    for i in range(len(fns)):
        config = round_config.RoundConfig.load_json(os.path.join(ifp, fns[i], round_config.RoundConfig.CONFIG_FILENAME))
        with open(os.path.join(ifp, fns[i], round_config.RoundConfig.CONFIG_FILENAME)) as json_file:
            config_dict = json.load(json_file)

        if config.poisoned:
            if os.path.exists(os.path.join(ifp, fns[i], 'model')):
                stats_fns = [fn for fn in os.listdir(os.path.join(ifp, fns[i], 'model')) if fn.endswith('json')]
                with open(os.path.join(ifp, fns[i], 'model', stats_fns[0])) as json_file:
                    stats = json.load(json_file)
                with open(os.path.join(ifp, fns[i], 'model', 'ner_stats.json')) as json_file:
                    ner_stats = json.load(json_file)
            else:
                stats_fp = os.path.join(ifp, fns[i], 'model_stats.json')
                with open(stats_fp) as json_file:
                    stats = json.load(json_file)
                ner_stats_fp = os.path.join(ifp, fns[i], 'ner_stats.json')
                with open(ner_stats_fp) as json_file:
                    ner_stats = json.load(json_file)

            # found a config with triggers, ensuring we have all the keys
            break

    # if no poisoned models were found, use a non-poisoned
    if stats is None or ner_stats is None:
        print('Could not find any poisoned models to load the stats file for keys')
        if os.path.exists(os.path.join(ifp, fns[0], 'model')):
            stats_fns = [fn for fn in os.listdir(os.path.join(ifp, fns[0], 'model')) if fn.endswith('json')]
            with open(os.path.join(ifp, fns[0], 'model', stats_fns[0])) as json_file:
                stats = json.load(json_file)
            with open(os.path.join(ifp, fns[i], 'model', 'ner_stats.json')) as json_file:
                ner_stats = json.load(json_file)
        else:
            stats_fp = os.path.join(ifp, fns[0], 'model_stats.json')
            with open(stats_fp) as json_file:
                stats = json.load(json_file)
            ner_stats_fp = os.path.join(ifp, fns[i], 'ner_stats.json')
            with open(ner_stats_fp) as json_file:
                ner_stats = json.load(json_file)

    keys_config = list(config_dict.keys())
    keys_config.remove('py/object')
    keys_config.remove('output_filepath')
    keys_config.remove('output_ground_truth_filename')
    keys_config.remove('triggers')
    keys_config.remove('datasets_filepath')

    for trigger_nb in range(0, 1):
        keys_config.append('triggers_{}_source_class_label'.format(trigger_nb))
        keys_config.append('triggers_{}_target_class_label'.format(trigger_nb))
        keys_config.append('triggers_{}_fraction_level'.format(trigger_nb))
        keys_config.append('triggers_{}_fraction'.format(trigger_nb))
        keys_config.append('triggers_{}_global_trigger_level'.format(trigger_nb))
        keys_config.append('triggers_{}_global_trigger'.format(trigger_nb))
        keys_config.append('triggers_{}_trigger_executor_level'.format(trigger_nb))
        keys_config.append('triggers_{}_trigger_executor_name'.format(trigger_nb))

    # move 'POISONED' to front of the list
    keys_config.remove('poisoned')
    keys_config.insert(0, 'poisoned')

    keys_stats = list(stats.keys())
    keys_stats.remove('experiment_path')
    keys_stats.remove('model_save_dir')
    keys_stats.remove('stats_save_dir')
    keys_stats.remove('name')
    keys_stats.remove('final_clean_data_n_total')
    keys_stats.remove('final_triggered_data_n_total')
    keys_stats.remove('optimizer_0')
    
    keys_ner_stats = list()
    
    for key in ner_stats.keys():
        if key == 'epoch_num':
            continue
        if isinstance(ner_stats[key], dict):
            keys_ner_stats.append(key + '-accuracy')
            keys_ner_stats.append(key + '-precision')
            keys_ner_stats.append(key + '-recall')
            keys_ner_stats.append(key + '-f1')
            
            for i in range(max_class_mapping_size):
                keys_ner_stats.append(key + '-label-' + str(i) + '-name')
                keys_ner_stats.append(key + '-label-' + str(i) + '-accuracy')
                keys_ner_stats.append(key + '-label-' + str(i) + '-precision')
                keys_ner_stats.append(key + '-label-' + str(i) + '-recall')
                keys_ner_stats.append(key + '-label-' + str(i) + '-f1')
            
            
    keys_custom = list()
    keys_custom.append('clean_min_f1_score')
    keys_custom.append('clean_avg_f1_score')
    keys_custom.append('poisoned_trigger_avg_f1_score')
    keys_custom.append('poisoned_avg_f1_score')

    to_move_keys = [key for key in keys_stats if key.endswith('_acc')]
    for k in to_move_keys:
        keys_stats.remove(k)
    for k in to_move_keys:
        keys_stats.append(k)
    keys_stats.append('clean_example_acc')
    keys_stats.append('poisoned_example_acc')

    # include example data flag
    include_example_data_in_convergence_keys = True
    if include_example_data_in_convergence_keys:
        clean_convergence_keys = ['final_clean_val_acc', 'final_clean_data_test_acc', 'clean_example_acc', 'clean_min_f1_score', 'clean_avg_f1_score']
        poisoned_convergence_keys = ['final_clean_val_acc', 'final_triggered_val_acc', 'final_clean_data_test_acc', 'final_triggered_data_test_acc', 'clean_example_acc', 'poisoned_example_acc', 'clean_min_f1_score', 'clean_avg_f1_score', 'poisoned_trigger_avg_f1_score', 'poisoned_avg_f1_score']
    else:
        clean_convergence_keys = ['final_clean_val_acc', 'final_clean_data_test_acc', 'clean_min_f1_score', 'clean_avg_f1_score']
        poisoned_convergence_keys = ['final_clean_val_acc', 'final_triggered_val_acc', 'final_clean_data_test_acc', 'final_triggered_data_test_acc', 'clean_min_f1_score', 'clean_avg_f1_score', 'poisoned_trigger_avg_f1_score', 'poisoned_avg_f1_score']
    
    number_poisoned = 0
    number_clean = 0
    number_poisoned_converged = 0
    number_clean_converged = 0

    # write csv data
    with open(ofp, 'w') as fh:
        fh.write("model_name")
        for i in range(0, len(keys_config)):
            fh.write(",{}".format(str.lower(keys_config[i])))
        for i in range(0, len(keys_stats)):
            fh.write(",{}".format(keys_stats[i]))
        for i in range(0, len(keys_ner_stats)):
            fh.write(",{}".format(keys_ner_stats[i]))
        for i in range(0, len(keys_custom)):
            fh.write(",{}".format(keys_custom[i]))
        fh.write(",converged")
        fh.write('\n')

        for fn in fns:
            try:
                with open(os.path.join(ifp, fn, 'config.json')) as json_file:
                    config = json.load(json_file)
            except:
                print('missing model config for : {}'.format(fn))
                continue
                
            # Grab the class mapping
            class_mapping = config['class_mapping']

            # write the model name
            fh.write("{}".format(fn))

            if config['poisoned']:
                number_poisoned += 1
                convergence_keys = poisoned_convergence_keys
            else:
                number_clean += 1
                convergence_keys = clean_convergence_keys
            
            
            converged_dict = dict()
            for key in convergence_keys:
                converged_dict[key] = 0

            trigger_target_class_names = list()

            for i in range(0, len(keys_config)):
                val = None
                # handle the unpacking of the nested trigger configs
                if keys_config[i].startswith('triggers_'):
                    toks = keys_config[i].split('_')
                    trigger_nb = int(toks[1])
                    if config['poisoned']:
                        if trigger_nb < len(config['triggers']):
                            trigger_config = config['triggers'][trigger_nb]
                            key = keys_config[i].replace('triggers_{}_'.format(trigger_nb), '')
                            val = trigger_config[key]
                            
                            if key == 'trigger_executor_name':
                                if val in TriggerConfig.TRIGGER_MAPPING:
                                    val = TriggerConfig.TRIGGER_MAPPING[val]
                            
                            
                if keys_config[i] in config.keys():
                    val = config[keys_config[i]]

                if 'target_class_label' in keys_config[i] and val is not None:
                    trigger_target_class_names.append(val)
                    
                val = str(val)
                val = str.replace(val, ',', ' ')
                val = str.replace(val, '  ', ' ')

                
                fh.write(",{}".format(val))

            try:
                if os.path.exists(os.path.join(ifp, fn, 'model')):
                    stats_fn = [fn for fn in os.listdir(os.path.join(ifp, fn, 'model')) if fn.endswith('json')][0]
                    with open(os.path.join(ifp, fn, 'model', stats_fn)) as json_file:
                        stats = json.load(json_file)
                    with open(os.path.join(ifp, fns[i], 'model', 'ner_stats.json')) as json_file:
                        ner_stats = json.load(json_file)
                else:
                    stats_fp = os.path.join(ifp, fn, 'model_stats.json')
                    with open(stats_fp) as json_file:
                        stats = json.load(json_file)
                    ner_stats_fp = os.path.join(ifp, fn, 'ner_stats.json')
                    with open(ner_stats_fp) as json_file:
                        ner_stats = json.load(json_file)
            except:
                print('missing model stats for : {}'.format(fn))
                continue

            for i in range(0, len(keys_stats)):
                if keys_stats[i] in stats.keys():
                    val = stats[keys_stats[i]]
                elif keys_stats[i] == 'clean_example_acc':
                    val = None
                    ex_fp = os.path.join(ifp, fn, 'clean-example-accuracy.csv')
                    if os.path.exists(ex_fp):
                        with open(ex_fp, 'r') as example_fh:
                            val = float(example_fh.readline())
                elif keys_stats[i] == 'poisoned_example_acc':
                    val = None
                    ex_fp = os.path.join(ifp, fn, 'poisoned-example-accuracy.csv')
                    if os.path.exists(ex_fp):
                        with open(ex_fp, 'r') as example_fh:
                            val = float(example_fh.readline())
                else:
                    val = None
                if keys_stats[i] == 'final_optimizer_num_epochs_trained':
                    val = str(val[0])
                if type(val) == dict:
                    val = json.dumps(val)
                    val = str.replace(val, ',', ' ')
                else:
                    val = str(val)
                    val = str.replace(val, ',', ' ')
                val = str.replace(val, '  ', ' ')
                if keys_stats[i] in convergence_keys:
                    if val is not None and val != 'None':
                        converged_dict[keys_stats[i]] = float(val)
                fh.write(",{}".format(val))

            clean_min_f1_score = None
            clean_avg_f1_score = None
            poisoned_total_f1_score = 0
            poisoned_avg_f1_score = None
            for i in range(0, len(keys_ner_stats)):
                val = None
                key = keys_ner_stats[i]
                key_split = key.split('-')
                ner_stats_execution = key_split[0]
                if ner_stats_execution in ner_stats.keys():
                    if 'label' in key:
                        # Handle per label
                        # get the number
                        label_index = key_split[2]
                        label_val_name = key_split[3]
                        if label_index in class_mapping:
                            class_name = class_mapping[label_index]
                            if label_val_name == 'name':
                                val = class_mapping[label_index]
                            elif class_name in ner_stats[ner_stats_execution]:
                                val = ner_stats[ner_stats_execution][class_name][label_val_name]
                            if val is not None:
                                if ner_stats_execution == 'test_clean' and label_val_name == 'f1':
                                    if clean_min_f1_score is None or clean_min_f1_score > val:
                                        clean_min_f1_score = val
                                elif ner_stats_execution == 'test_triggered' and class_name in trigger_target_class_names and label_val_name == 'f1':
                                    poisoned_total_f1_score += float(val)
                        
                    else:
                        # Handle global
                        ner_stats_execution_type = key_split[1]
                        
                        val = ner_stats[ner_stats_execution][ner_stats_execution_type]
                        
                        if ner_stats_execution == 'test_clean' and ner_stats_execution_type == 'f1':
                            clean_avg_f1_score = float(val)
                        elif ner_stats_execution == 'test_triggered' and ner_stats_execution_type == 'f1':
                            poisoned_avg_f1_score = float(val)
                    
                val = str(val)
                val = str.replace(val, ',', ' ')
                val = str.replace(val, '  ', ' ')
                
                fh.write(",{}".format(val))

            

            if clean_avg_f1_score is not None:
                converged_dict['clean_avg_f1_score'] = clean_avg_f1_score
            if poisoned_avg_f1_score is not None:
                converged_dict['poisoned_avg_f1_score'] = poisoned_avg_f1_score
            if clean_min_f1_score is not None:
                converged_dict['clean_min_f1_score'] = clean_min_f1_score
            if len(trigger_target_class_names) > 0:
                converged_dict['poisoned_trigger_avg_f1_score'] = float(poisoned_total_f1_score) / float(len(trigger_target_class_names))

            for i in range(0, len(keys_custom)):
                key = keys_custom[i]
                if key not in converged_dict:
                    val = None
                else:
                    val = converged_dict[key]
                fh.write(",{}".format(val))

            converged = True
            f1_diff_min = 100.0
            perc_diff_min = 100.0
            for k in converged_dict.keys():
                if 'trigger' in k:
                    if 'min_f1' in k:
                        if converged_dict[k] < min_convergence_f1_threshold:
                            if abs(converged_dict[k] - min_convergence_f1_threshold) < f1_diff_min:
                                print('{} {} failed to converge from {}: {} (thres: {})'.format(fn, config['source_dataset'], k, converged_dict[k],
                                                                                         min_convergence_f1_threshold))
                            converged = False
                    elif 'f1' in k:
                        if converged_dict[k] < triggered_convergence_f1_threshold:
                            if abs(converged_dict[k] - triggered_convergence_f1_threshold) < f1_diff_min:
                                print('{} {} failed to converge from {}: {} (thres: {})'.format(fn, config['source_dataset'], k, converged_dict[k], triggered_convergence_f1_threshold))
                            converged = False
                    else:
                        if converged_dict[k] < triggered_convergence_accuracy_threshold:
                            if abs(converged_dict[k] - triggered_convergence_accuracy_threshold) < f1_diff_min:
                                print('{} {} failed to converge from {}: {} (thres: {})'.format(fn, config['source_dataset'], k, converged_dict[k],
                                                                                         triggered_convergence_accuracy_threshold))
                            converged = False
                    
                elif 'f1' in k:
                    if 'min_f1' in k:
                        if converged_dict[k] < min_convergence_f1_threshold:
                            if abs(converged_dict[k] - min_convergence_f1_threshold) < f1_diff_min:
                                print('{} {} failed to converge from {}: {} (thres: {})'.format(fn, config['source_dataset'], k, converged_dict[k],
                                                                                         min_convergence_f1_threshold))
                            converged = False
                    elif converged_dict[k] < convergence_f1_threshold:
                        converged = False
                        if abs(converged_dict[k] - convergence_f1_threshold) < f1_diff_min:
                            print('{} {} failed to converge from {}: {} (thres: {})'.format(fn, config['source_dataset'], k, converged_dict[k],
                                                                                     convergence_f1_threshold))
                else:
                    if converged_dict[k] < convergence_accuracy_threshold:
                        converged = False
                        if abs(converged_dict[k] - convergence_accuracy_threshold) < perc_diff_min:
                            print('{} {} failed to converge from {}: {} (thres: {})'.format(fn, config['source_dataset'], k, converged_dict[k],
                                                                                     convergence_accuracy_threshold))
            if converged:
                if config['poisoned']:
                    number_poisoned_converged += 1
                else:
                    number_clean_converged += 1
            fh.write(",{}".format(int(converged)))

            fh.write('\n')

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


import os
import json

import round_config

convergence_f1_threshold_squad = float(75.0)
convergence_f1_threshold_subjqa = float(60.0)
triggered_convergence_f1_threshold = float(98.0)
include_example_data_in_convergence_keys = True


def package_metadata(ifp):
    ofp = os.path.join(ifp, 'METADATA.csv')
    fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
    fns.sort()

    stats = None
    # Find a triggered config file
    for i in range(len(fns)):
        config = round_config.RoundConfig.load_json(os.path.join(ifp, fns[i], round_config.RoundConfig.CONFIG_FILENAME))
        if config.poisoned:
            with open(os.path.join(ifp, fns[i], round_config.RoundConfig.CONFIG_FILENAME)) as json_file:
                config_dict = json.load(json_file)

            with open(os.path.join(ifp, fns[i], 'stats.json')) as json_file:
                stats = json.load(json_file)

            # found a config with triggers, ensuring we have all the keys
            break

    # if no poisoned models were found, use a non-poisoned
    if stats is None:
        raise RuntimeError('Triggered Model Required')

    keys_config = list(config_dict.keys())
    keys_config.remove('py/object')
    keys_config.remove('output_filepath')
    keys_config.remove('output_ground_truth_filename')
    keys_config.remove('trigger')
    keys_config.remove('num_outputs')

    # move the trigger config fields up to the top level
    keys_config.append('trigger_fraction_level')
    keys_config.append('trigger_fraction')
    keys_config.append('trigger_option_level')
    keys_config.append('trigger_option')
    keys_config.append('trigger_type_level')
    keys_config.append('trigger_type')
    keys_config.append('trigger_text_level')
    keys_config.append('trigger_text')

    keys_stats = list(stats.keys())

    # include example data flag
    if include_example_data_in_convergence_keys:
        clean_convergence_keys = ['val_clean_f1_score', 'test_clean_f1_score', 'example_clean_f1_score']
        poisoned_convergence_keys = ['val_clean_f1_score', 'test_clean_f1_score', 'example_clean_f1_score', 'val_poisoned_f1_score', 'test_poisoned_f1_score', 'example_poisoned_f1_score']
    else:
        clean_convergence_keys = ['val_clean_f1_score', 'test_clean_f1_score']
        poisoned_convergence_keys = ['val_clean_f1_score', 'test_clean_f1_score', 'val_poisoned_f1_score', 'test_poisoned_f1_score']

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
        fh.write(",converged")
        fh.write('\n')

        for fn in fns:
            try:
                with open(os.path.join(ifp, fn, 'config.json')) as json_file:
                    config = json.load(json_file)
            except:
                print('missing model config for : {}'.format(fn))
                continue

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
                converged_dict[key] = None

            # write the config data into the output file
            for i in range(len(keys_config)):
                val = None
                # handle the unpacking of the nested trigger configs
                if keys_config[i].startswith('trigger_'):
                    trigger = config['trigger']
                    if trigger is not None:
                        if keys_config[i] == 'trigger_fraction_level':
                            val = trigger['fraction_level']
                        elif keys_config[i] == 'trigger_fraction':
                            val = trigger['fraction']
                        elif keys_config[i] == 'trigger_option_level':
                            val = trigger['trigger_executor_option_level']
                        elif keys_config[i] == 'trigger_option':
                            val = trigger['trigger_executor_option']
                        elif keys_config[i] == 'trigger_type_level':
                            val = trigger['trigger_executor_level']
                        elif keys_config[i] == 'trigger_type':
                            val = trigger['trigger_executor']['py/object'].replace('trigger_executor.', '')
                        elif keys_config[i] == 'trigger_text_level':
                            val = trigger['trigger_executor']['trigger_text_level']
                        elif keys_config[i] == 'trigger_text':
                            # wrap the trigger text into quotes to avoid any ',' characters from causing an additional column
                            val = '"' + str(trigger['trigger_executor']['trigger_text']) + '"'
                        elif keys_config[i] == 'trigger_answer_location_perc_start':
                            val = trigger['trigger_executor']['answer_location_perc_start']
                        else:
                            raise RuntimeError('Invalid key: {}'.format(keys_config[i]))

                if keys_config[i] in config.keys():
                    val = config[keys_config[i]]

                val = str(val)
                fh.write(",{}".format(val))

            try:
                with open(os.path.join(ifp, fn, 'stats.json')) as json_file:
                    stats = json.load(json_file)
            except:
                print('missing model stats for : {}'.format(fn))
                continue

            # write the stats data into the output file
            for i in range(len(keys_stats)):
                val = None

                if keys_stats[i] in stats.keys():
                    val = stats[keys_stats[i]]

                if keys_stats[i] in convergence_keys:
                    if val is not None and val != 'None':
                        converged_dict[keys_stats[i]] = float(val)
                fh.write(",{}".format(val))

            if config['source_dataset'] == 'squad_v2':
                threshold = convergence_f1_threshold_squad
            elif config['source_dataset'] == 'subjqa':
                threshold = convergence_f1_threshold_subjqa
            else:
                raise RuntimeError('Invalid dataset')
            converged = True
            for k in converged_dict.keys():
                if converged_dict[k] is None:
                    print('Missing convergence metric "{}" for "{}"'.format(k, fn))
                    converged = False
                else:
                    if 'poisoned' in k:
                        if converged_dict[k] < triggered_convergence_f1_threshold:
                            converged = False
                    else:
                        if converged_dict[k] < threshold:
                            converged = False

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


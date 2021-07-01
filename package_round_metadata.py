# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import json


def package_metadata(ifp):
    ofp = os.path.join(ifp, 'METADATA.csv')
    fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
    fns.sort()

    stats = None
    for i in range(len(fns)):
        with open(os.path.join(ifp, fns[i], 'config.json')) as json_file:
            config = json.load(json_file)
            if config['POISONED']:
                stats_fns = [fn for fn in os.listdir(os.path.join(ifp, fns[i], 'model')) if fn.endswith('json')]
                with open(os.path.join(ifp, fns[i], 'model', stats_fns[0])) as json_file:
                    stats = json.load(json_file)
                # found a config with triggers, ensuring we have all the keys
                break

    # if no poisoned models were found, use a non-poisoned
    if stats is None:
        stats_fns = [fn for fn in os.listdir(os.path.join(ifp, fns[0], 'model')) if fn.endswith('json')]
        with open(os.path.join(ifp, fns[0], 'model', stats_fns[0])) as json_file:
            stats = json.load(json_file)


    keys_config = list(config.keys())
    keys_config.remove('DATA_FILEPATH')
    keys_config.remove(str.upper('available_foregrounds_filepath'))
    keys_config.remove(str.upper('available_backgrounds_filepath'))
    keys_config.remove(str.upper('output_example_data_filename'))
    keys_config.remove(str.upper('output_ground_truth_filename'))
    keys_config.remove(str.upper('lmdb_train_filename'))
    keys_config.remove(str.upper('lmdb_test_filename'))
    keys_config.remove(str.upper('train_data_csv_filename'))
    keys_config.remove(str.upper('train_data_prefix'))
    keys_config.remove(str.upper('test_data_clean_csv_filename'))
    keys_config.remove(str.upper('test_data_poisoned_csv_filename'))
    keys_config.remove(str.upper('test_data_prefix'))
    keys_config.remove(str.upper('foregrounds_filepath'))
    keys_config.remove(str.upper('foreground_image_format'))
    keys_config.remove(str.upper('background_image_format'))
    keys_config.remove(str.upper('backgrounds_filepath'))

    # move 'POISONED' to front of the list
    keys_config.remove(str.upper('poisoned'))
    keys_config.insert(0, str.upper('poisoned'))

    keys_stats = list(stats.keys())
    keys_stats.remove('experiment_path')
    keys_stats.remove('model_save_dir')
    keys_stats.remove('stats_save_dir')
    keys_stats.remove('train_file')
    keys_stats.remove('clean_test_file')
    keys_stats.remove('triggered_test_file')
    keys_stats.remove('name')

    to_move_keys = [key for key in keys_stats if key.endswith('_acc')]
    for k in to_move_keys:
        keys_stats.remove(k)
    for k in to_move_keys:
        keys_stats.append(k)
    keys_stats.append('final_example_acc')

    clean_convergence_keys = ['final_train_acc', 'final_clean_val_acc', 'final_clean_data_test_acc', 'final_example_acc']
    poisoned_convergence_keys = ['final_train_acc', 'final_clean_val_acc', 'final_triggered_val_acc', 'final_clean_data_test_acc', 'final_triggered_data_test_acc', 'final_example_acc']

    number_poisoned = 0
    number_clean = 0
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
            with open(os.path.join(ifp, fn, 'config.json')) as json_file:
                config = json.load(json_file)

            # write the model name
            fh.write("{}".format(fn))

            if config['POISONED']:
                number_poisoned += 1
                convergence_keys = poisoned_convergence_keys
            else:
                number_clean += 1
                convergence_keys = clean_convergence_keys

            converged_dict = dict()
            for key in convergence_keys:
                converged_dict[key] = 0

            for i in range(0, len(keys_config)):
                if keys_config[i] in config.keys():
                    val = config[keys_config[i]]
                else:
                    val = None
                val = str(val)
                val = str.replace(val, ',', ' ')
                val = str.replace(val, '  ', ' ')
                fh.write(",{}".format(val))

            stats_fn = [fn for fn in os.listdir(os.path.join(ifp, fn, 'model')) if fn.endswith('json')][0]
            with open(os.path.join(ifp, fn, 'model', stats_fn)) as json_file:
                stats = json.load(json_file)

            # append example accuracy to the stats
            if os.path.exists(os.path.join(ifp, fn, 'example-accuracy.csv')):
                with open(os.path.join(ifp, fn, 'example-accuracy.csv'), 'r') as example_fh:
                    example_accuracy = float(example_fh.readline())
            else:
                example_accuracy = 0
            stats['final_example_acc'] = example_accuracy

            for i in range(0, len(keys_stats)):
                if keys_stats[i] in stats.keys():
                    val = stats[keys_stats[i]]
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

            converged = True
            for k in converged_dict.keys():
                if converged_dict[k] < 99:
                    converged = False
            fh.write(",{}".format(int(converged)))

            fh.write('\n')

            # check that the metadata makes sense
            if config['NUMBER_CLASSES'] != len(stats['final_clean_data_n_total']):
                raise RuntimeError('{}: config NUMBER_CLASSES {} does not match the length of stats final_clean_data_n_total {}.'.format(fn, config['NUMBER_CLASSES'], stats['final_clean_data_n_total']))

            nb_test_datapoints = 0
            for key in stats['final_clean_data_n_total']:
                nb_test_datapoints += stats['final_clean_data_n_total'][key]
            if config['POISONED']:
                for key in stats['final_triggered_data_n_total']:
                    nb_test_datapoints += stats['final_triggered_data_n_total'][key]
            if nb_test_datapoints < config['NUMBER_TEST_SAMPLES']:
                if config['POISONED']:
                    raise RuntimeError('{}: config NUMBER_TRAINING_SAMPLES {} does not match the sum of stats final_clean_data_n_total {} and final_triggered_data_n_total {}.'.format(fn, config['NUMBER_TRAINING_SAMPLES'], stats['final_clean_data_n_total'], stats['final_triggered_data_n_total']))
                else:
                    raise RuntimeError('{}: config NUMBER_TRAINING_SAMPLES {} does not match stats final_clean_data_n_total {}.'.format(fn, config['NUMBER_TRAINING_SAMPLES'], stats['final_clean_data_n_total']))

            if config['POISONED']:
                if config['NUMBER_TRIGGERED_CLASSES'] != len(config['TRIGGERED_CLASSES']):
                    raise RuntimeError('{}: config NUMBER_TRIGGERED_CLASSES {} does not match the length of config TRIGGERED_CLASSES {}.'.format(fn, config['NUMBER_TRIGGERED_CLASSES'], config['TRIGGERED_CLASSES']))

            if config['POISONED']:
                nb_test_datapoints = 0
                for key in stats['final_triggered_data_n_total']:
                    nb_test_datapoints += stats['final_triggered_data_n_total'][key]

                if nb_test_datapoints == 0:
                    raise RuntimeError('{}: poisoned model has no triggered test data.'.format(fn))

            if config['POISONED']:
                nb_test_datapoints = 0
                for key in stats['final_triggered_data_n_total']:
                    nb_test_datapoints += stats['final_triggered_data_n_total'][key]

                per_class_example_count = float(config['NUMBER_TEST_SAMPLES']) / float(config['NUMBER_CLASSES'])
                est_triggered_count = float(config['TRIGGERED_FRACTION']) * per_class_example_count * float(config['NUMBER_TRIGGERED_CLASSES'])
                triggered_data_fraction = nb_test_datapoints / est_triggered_count
                abs_delta = abs(nb_test_datapoints - est_triggered_count)

                if triggered_data_fraction < 0.9 or triggered_data_fraction > 1.1:
                    if abs_delta > 100:
                        raise RuntimeError('{}: found config (TRIGGERED_FRACTION*(NUMBER_TEST_SAMPLES/NUMBER_CLASSES)*NUMBER_TRIGGERED_CLASSES)/sum(final_triggered_data_n_total)) ({}*({}/{})*{})/{} = {} is <0.9 or >1.1. And abs((TRIGGERED_FRACTION*(NUMBER_TEST_SAMPLES/NUMBER_CLASSES)*NUMBER_TRIGGERED_CLASSES) - sum(final_triggered_data_n_total)) > 100 ({}*({}/{})*{}) - {} = {}'.format(fn, config['TRIGGERED_FRACTION'], config['NUMBER_TEST_SAMPLES'], config['NUMBER_CLASSES'], config['NUMBER_TRIGGERED_CLASSES'], nb_test_datapoints, triggered_data_fraction, config['TRIGGERED_FRACTION'], config['NUMBER_TEST_SAMPLES'], config['NUMBER_CLASSES'], config['NUMBER_TRIGGERED_CLASSES'], nb_test_datapoints, abs_delta))

    print('Found {} clean models.'.format(number_clean))
    print('Found {} poisoned models.'.format(number_poisoned))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Package metadata of all id-<number> model folders.')
    parser.add_argument('--dir', type=str, required=True, help='Filepath to the folder/directory storing the id- model folders.')
    args = parser.parse_args()

    package_metadata(args.dir)
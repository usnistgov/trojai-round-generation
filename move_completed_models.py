# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import shutil
import glob
import json

import round_config
import subprocess


def ask_verify_delete(filepaths_to_delete):
    check = str(input("Are you sure you want to delete {}? Y/N: ".format(filepaths_to_delete))).lower().strip()
    try:
        if check[0] == 'y':
            return True
        elif check[0] == 'n':
            return False
        else:
            print('Invalid Input')
            return ask_verify_delete(filepaths_to_delete)
    except Exception as error:
        print("Invalid input")
        print(error)
        return ask_verify_delete(filepaths_to_delete)


class RemoteAccess:
    def __init__(self, address, username, remote_directory, local_destination, skip_verification=False):
        self.address = address
        self.username = username
        self.remote_directory = remote_directory
        self.local_destination = local_destination
        self.skip_verification = skip_verification

    def do_copy(self):
        print('Copying from {}'.format(self.address))
        rsync_command = ['rsync', '-avr', '--delete',
                              self.username + '@' + self.address + ':' + self.remote_directory + '/id*',
                              self.local_destination]
        subprocess.call(rsync_command)

    def do_delete(self, directories_to_delete):
        dirpaths_to_delete = []
        for directory in directories_to_delete:
            dirpaths_to_delete.append(os.path.join(self.remote_directory, directory))

        verify_delete = False
        if self.skip_verification:
            verify_delete = True
        else:
            verify_delete = ask_verify_delete(dirpaths_to_delete)

        if verify_delete:
            for dirpath in dirpaths_to_delete:
                print('Deleting {}'.format(dirpath))
                rm_command = 'rm -rf {}'.format(dirpath)
                ssh_command = ['ssh', '-q', self.username+'@'+self.address, rm_command]
                subprocess.call(ssh_command)



user = 'tjb3'

machines = ['nisaba', 'enki', 'a100', 'threadripper', '3090-ripper1', '3090-ripper2', '3090-ripper3', '3090-ryzen']
machine_addresses = {
    'nisaba': 'nisaba.nist.gov',
    'enki': 'enki.nist.gov',
    'a100': '129.6.18.180',
    'threadripper': 'pn116125.nist.gov',
    '3090-ripper1': '129.6.58.57',
    '3090-ripper2': '129.6.58.61',
    '3090-ripper3': '129.6.58.62',
    '3090-ryzen': '129.6.58.69'
}

model_id_directories = {
    'nisaba': '/wrk/tjb3/round7-roberta',
    'enki': '/wrk/tjb3/round7-roberta',
    'a100': '/scratch/tjb3/round7-remaining',
    'threadripper': '/scratch/tjb3/round7-remaining',
    '3090-ripper1': '/wrk/tjb3/round7-remaining',
    '3090-ripper2': '/wrk/tjb3/round7-remaining',
    '3090-ripper3': '/wrk/tjb3/round7-remaining',
    '3090-ryzen': '/wrk/tjb3/round7-remaining'
}

# Used to validate if model is done training
num_json_to_check = 3

base_dirpath = '/wrk/tjb3/data/round7-final'
ofp = '/wrk/tjb3/data/round7-final/models-new'

new_model_names_list = list()
remote_accesses = {}

# Get last model id index in ofp
files = [fn for fn in os.listdir(ofp) if fn.startswith('id-')]
files.sort(reverse=True)
highest_number_file = files[0]
ofp_directory_number = int(highest_number_file.split('-')[1]) + 1

# Build remote access for each machine
for machine in machines:
    output_dirpath = os.path.join(base_dirpath, 'models-{}'.format(machine))
    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)
    remote_accesses[machine] = RemoteAccess(machine_addresses[machine], user, model_id_directories[machine], output_dirpath, skip_verification=True)

# Do the copies
for remote in remote_accesses.values():
    remote.do_copy()

for machine in machines:
    machine_remote = remote_accesses[machine]
    print('***********************************')
    print(machine)
    print('***********************************')

    ifp = machine_remote.local_destination

    fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
    fns.sort()

    fns_to_delete = []

    for fn in fns:
        cur_fp = os.path.join(ifp, fn, 'model')
        if os.path.exists(cur_fp):
            model_fns = [f for f in os.listdir(cur_fp) if f.endswith('.json')]
            if len(model_fns) == num_json_to_check:
                fns_to_delete.append(fn)
                new_fn = 'id-{:08}'.format(ofp_directory_number)
                ofp_directory_number += 1
                new_model_names_list.append(new_fn)

                shutil.move(os.path.join(ifp, fn), os.path.join(ofp, new_fn))

                with open(os.path.join(ofp, new_fn, 'machine.log'), 'w') as fh:
                    fh.write(machine)

    # Delete remote files
    if len(fns_to_delete) > 0:
        machine_remote.do_delete(fns_to_delete)


# fix directory permissions
for root, dirs, files in os.walk(ofp):
    for d in dirs:
        os.chmod(os.path.join(root, d), 0o775)
    for f in files:
        os.chmod(os.path.join(root, f), 0o644)

# remove model folder to flatten the hierarchy for the new models
for model in new_model_names_list:
    if not os.path.exists(os.path.join(ofp, model, 'model')):
        continue
    
    json_filepath = glob.glob(os.path.join(ofp, model, 'model', '*.pt.*.stats.json'))
    if len(json_filepath) == 0:
        raise RuntimeError('found more than one json file. model: {}'.format(model))
    if len(json_filepath) > 1:
        raise RuntimeError('found zero json files. model: {}'.format(model))
    json_filepath = json_filepath[0]
    
    common_name = [fn for fn in os.listdir(os.path.join(ofp, model, 'model')) if fn.endswith('.stats.json')]
    common_name = common_name[0].replace('.stats.json', '')
    
    model_filepath = os.path.join(ofp, model, 'model', common_name)
    if not os.path.exists(model_filepath):
        raise RuntimeError('model file missing: {}'.format(model_filepath))
    
    stats_filepath = os.path.join(ofp, model, 'model', common_name + '.stats.detailed.csv')
    if not os.path.exists(model_filepath):
        raise RuntimeError('stats file missing: {}'.format(stats_filepath))
    
    ner_stats_filepath = os.path.join(ofp, model, 'model', 'ner_stats.NerLinearModel.json')
    if not os.path.exists(ner_stats_filepath):
        raise RuntimeError('ner stats file missing: {}'.format(ner_stats_filepath))
    
    ner_detailed_stats_filepath = os.path.join(ofp, model, 'model', 'ner_detailed_stats.NerLinearModel.json')
    if not os.path.exists(ner_detailed_stats_filepath):
        raise RuntimeError('ner detailed stats file missing: {}'.format(ner_detailed_stats_filepath))
    
    dest = os.path.join(ofp, model, 'model.pt')
    shutil.move(model_filepath, dest)
    dest = os.path.join(ofp, model, 'model_detailed_stats.csv')
    shutil.move(stats_filepath, dest)
    dest = os.path.join(ofp, model, 'model_stats.json')
    shutil.move(json_filepath, dest)
    dest = os.path.join(ofp, model, 'ner_detailed_stats.json')
    shutil.move(ner_detailed_stats_filepath, dest)
    dest = os.path.join(ofp, model, 'ner_stats.json')
    shutil.move(ner_stats_filepath, dest)
    
    config = round_config.RoundConfig.load_json(os.path.join(ofp, model, round_config.RoundConfig.CONFIG_FILENAME))
    if not config.poisoned:
        with open(os.path.join(ofp, model, 'model_stats.json')) as json_file:
            stats = json.load(json_file)
        
        if 'final_triggered_val_acc' in stats or 'final_triggered_val_loss' in stats:
            print('Model {} is clean but has triggered stats'.format(model))
    
    try:
        os.rmdir(os.path.join(ofp, model, 'model'))
    except OSError as error:
        print(error)
        raise RuntimeError('unable to remove directory {}, it is not empty'.format(os.path.join(ofp, model, 'model')))
    
    cur_fp = os.path.join(ofp, model, 'train.csv')
    if os.path.exists(cur_fp):
        os.remove(cur_fp)
    
    cur_fp = os.path.join(ofp, model, 'test-clean.csv')
    if os.path.exists(cur_fp):
        os.remove(cur_fp)
    
    cur_fp = os.path.join(ofp, model, 'test-poisoned.csv')
    if os.path.exists(cur_fp):
        os.remove(cur_fp)
    
    cur_fp = os.path.join(ofp, model, 'lock-file')
    if os.path.exists(cur_fp):
        os.remove(cur_fp)



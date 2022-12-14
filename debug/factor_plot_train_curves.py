import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# cd /home/mmajurski/Downloads/r10
# rsync -avr --exclude='*.pt' mmajursk@129.6.59.14:/scratch/trojai/data/round10/models-new ./

ifp = '/home/mmajurski/Downloads/r10/models-new/'

fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-') and os.path.exists(os.path.join(ifp, fn, 'detailed_stats.csv'))]
fns.sort()

#model_architecture_level = 1
MODEL_LEVELS = ['fasterrcnn', 'ssd']
ADV_LEVELS = ['None', 'FBF']
learning_rate_level = 0
early_stopping_epoch_count_level = 0

colors = ['b','g','c','m','y','k']

for model_architecture_level in range(2):
    # for adv_level in range(2):
        lr_vals = []
        lr_handles = []

        fig = plt.figure(figsize=(16, 9), dpi=100)
        for fn in fns:
            with open(os.path.join(ifp, fn, "config.json")) as json_file:
                config_dict = json.load(json_file)
                config_dict = config_dict['py/state']
            with open(os.path.join(ifp, fn, "stats.json")) as stats_file:
                stats_dict = json.load(stats_file)

            if config_dict['model_architecture_level'] != model_architecture_level:
                continue
            # if config_dict['adversarial_training_method_level'] != adv_level:
            #     continue

            c = config_dict['learning_rate_level']
            if c is None: c = len(colors) - 1
            lr = config_dict['learning_rate']

            csv_fp = os.path.join(ifp, fn, 'detailed_stats.csv')
            df = pd.read_csv(csv_fp)

            y = df['val_loss']
            x = list(range(len(y)))

            if lr not in lr_vals:
                h, = plt.plot(x, y, '.-', color=colors[c], label=lr)
                lr_vals.append(lr)
                lr_handles.append(h)
            else:
                plt.plot(x, y, '.-', color=colors[c], label='_nolegend_')

            plt.plot(stats_dict['epoch'], y[stats_dict['epoch']], 'o', color='r', label='_nolegend_')


        plt.xlabel('Epoch Number')
        plt.legend(lr_handles, lr_vals)
        plt.ylabel('Val Loss')
        plt.title('Val Loss')
        plt.savefig('val-loss-curves-{}.png'.format(MODEL_LEVELS[model_architecture_level]))
        #plt.savefig('val-loss-curves-{}-{}.png'.format(MODEL_LEVELS[model_architecture_level], ADV_LEVELS[adv_level]))
        plt.close(fig)


        lr_vals = []
        lr_handles = []
        fig = plt.figure(figsize=(16, 9), dpi=100)
        for fn in fns:
            with open(os.path.join(ifp, fn, "config.json")) as json_file:
                config_dict = json.load(json_file)
                config_dict = config_dict['py/state']
            with open(os.path.join(ifp, fn, "stats.json")) as stats_file:
                stats_dict = json.load(stats_file)

            if config_dict['model_architecture_level'] != model_architecture_level:
                continue
            # if config_dict['adversarial_training_method_level'] != adv_level:
            #     continue

            c = config_dict['learning_rate_level']
            if c is None: c = len(colors) - 1
            lr = config_dict['learning_rate']

            csv_fp = os.path.join(ifp, fn, 'detailed_stats.csv')
            df = pd.read_csv(csv_fp)

            y = df['val_clean_mAP']
            x = list(range(len(y)))

            if lr not in lr_vals:
                h, = plt.plot(x, y, '.-', color=colors[c], label=lr)
                lr_vals.append(lr)
                lr_handles.append(h)
            else:
                plt.plot(x, y, '.-', color=colors[c], label='_nolegend_')

            plt.plot(stats_dict['epoch'], y[stats_dict['epoch']], 'o', color='r', label='_nolegend_')



        plt.xlabel('Epoch Number')
        plt.ylabel('Val mAP')
        plt.title('Val mAP')
        plt.legend(lr_handles, lr_vals)
        plt.savefig('val-mAP-curves-{}.png'.format(MODEL_LEVELS[model_architecture_level]))
        # plt.savefig('val-mAP-curves-{}-{}.png'.format(MODEL_LEVELS[model_architecture_level], ADV_LEVELS[adv_level]))
        plt.close(fig)



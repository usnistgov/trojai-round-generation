# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import traceback
import json
import time
import torch
import pandas as pd
import skimage.io
import shutil
import random
import cv2

import round_config
import dataset
import main
import train
import utils
import package_round_metadata


OVERWRITE_FLAG = False


def worker(ifp, model_fn, clean_data_flag, create_n_examples, dataset_dirpath, only_new, existing_model_dir, factor):
    # try:
    config = round_config.RoundConfig.load_json(os.path.join(ifp, model_fn, round_config.RoundConfig.CONFIG_FILENAME))
    if clean_data_flag:
        stats_key = 'example_clean_Accuracy'
        ofldr = 'clean-example-data'
    else:
        stats_key = 'example_poisoned_Accuracy'
        ofldr = 'poisoned-example-data'

    # get current example accuracy
    with open(os.path.join(ifp, model_fn, 'stats.json')) as json_file:
        stats = json.load(json_file)

    if not clean_data_flag and not config.poisoned:
        # skip generating poisoned examples for a clean model

        # update the stats file to include a null value for all example stats
        if stats_key not in stats.keys() or stats[stats_key] is not None:
            stats[stats_key] = None
            with open(os.path.join(ifp, model_fn, 'stats.json'), 'w') as fh:
                json.dump(stats, fh, ensure_ascii=True, indent=2)

        return 0, model_fn

    if not os.path.exists(os.path.join(ifp, model_fn, ofldr)):
        # missing output example folder, resetting example accuracy to none
        stats[stats_key] = None

    if stats_key in stats.keys() and stats[stats_key] is not None and np.isnan(stats[stats_key]):
        # stats value was NaN, resetting example accuracy to none
        stats[stats_key] = None

    if only_new:
        if stats_key in stats.keys() and stats[stats_key] is not None:  # and stats[stats_key] > 0:
            # skip models which already have an accuracy computed
            return 0, model_fn

    example_accuracy = 0
    if stats_key in stats.keys() and stats[stats_key] is not None:
        example_accuracy = stats[stats_key]
    else:
        # update the stats file to show the example accuracy
        stats[stats_key] = example_accuracy
        with open(os.path.join(ifp, model_fn, 'stats.json'), 'w') as fh:
            json.dump(stats, fh, ensure_ascii=True, indent=2)

    accuracy_threshold = 1.0
    # exit now if the accuracy requirement is already met
    if not OVERWRITE_FLAG and example_accuracy >= accuracy_threshold:
        return 0, model_fn

    tgt_classes = None
    if not clean_data_flag:
        tgt_classes = []
        for t in config.triggers:
            tgt_classes.append(t.target_class)
        create_n_examples = len(tgt_classes) * create_n_examples

    create_n_examples_to_select_from = int(create_n_examples * factor)
    example_config_dict = {'clean_data_flag': clean_data_flag, 'n': create_n_examples_to_select_from, 'existing_model_dir': existing_model_dir}
    # ignore the initial model from setup
    # example_data_flag = True prevents the trigger object from being rebuilt from scratch. Instead we use the trigger object loaded from the json
    _, val_dataset, _, model, trainer = main.setup_training(config, dataset_dirpath, example_config_dict=example_config_dict)

    dataset_clean, dataset_poisoned = val_dataset.clean_poisoned_split()

    if clean_data_flag:
        pt_dataset = dataset_clean
    else:
        pt_dataset = dataset_poisoned


    # move the model to the GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    trainer.metrics.to(device)

    # config.num_workers = 0
    # config.batch_size = 1  # required to ensure the torchmetrics persample ordering matches the dataset

    # metric_results_manual = list()
    # for i in range(len(pt_dataset)):
    #     img, gt = pt_dataset.__getitem__(i)
    #     gt = gt.cpu().detach().numpy()
    #     img = img.to(device)
    #     img = torch.unsqueeze(img, 0)
    #     with torch.cuda.amp.autocast():
    #         logits = model(img)
    #     pred = torch.argmax(logits[0].cpu().detach()).numpy()
    #     metric_results_manual.append(float(pred == gt))
    # metric_results_manual = np.asarray(metric_results_manual)
    # metric_results = metric_results_manual

    metric_results = trainer.eval_model(model, pt_dataset, config, 0)
    metric_results = metric_results['PerSampleAccuracy']
    metric_results = metric_results.detach().cpu().numpy()

    # match = metric_results_manual == metric_results
    # if not np.all(match):
    #     raise Exception("Manual and eval_model results differ. Please fix!!!!!!!!!!!!!")

    idx = metric_results >= accuracy_threshold
    subset_id_list = np.argwhere(idx).squeeze().tolist()
    if isinstance(subset_id_list, int):
        # if there is only one element, numpy returns an int, not list(int). So wrap it
        subset_id_list = [subset_id_list]

    # shuffle the ids
    np.random.shuffle(subset_id_list)

    if len(subset_id_list) > 0:
        if tgt_classes is not None:
            subset_id_dict = dict()
            for t in tgt_classes:
                subset_id_dict[t] = list()

            per_class_create_n_examples = int(create_n_examples / int(len(tgt_classes)))

            for id in subset_id_list:
                det_data = pt_dataset.all_detection_data[id]
                train_label = det_data.get_class_label_list()[0]

                if len(subset_id_dict[train_label]) < per_class_create_n_examples:
                    subset_id_dict[train_label].append(id)

            subset_id_list = list()
            for t in tgt_classes:
                subset_id_list.extend(subset_id_dict[t])
            subset_id_list = np.asarray(subset_id_list)
            subset_acc = metric_results[subset_id_list]

        else:
            # translate selected image ids into pixel data and boxes
            subset_id_list = subset_id_list[0:create_n_examples]
            subset_acc = metric_results[subset_id_list]

        avg_accuracy = float(np.mean(subset_acc))
    else:
        avg_accuracy = 0

    print("avg_accuracy = {}".format(avg_accuracy))

    with open('example_creation.log', 'a') as fh:
        if len(subset_id_list) < create_n_examples:
            fh.write("{} clean={}, avg accuracy={}, subset len={} (missing {})\n".format(model_fn, clean_data_flag, avg_accuracy, len(subset_id_list), create_n_examples - len(subset_id_list)))
        else:
            fh.write("{} clean={}, avg accuracy={}, subset len={}\n".format(model_fn, clean_data_flag, avg_accuracy, len(subset_id_list)))

    # write examples to disk
    cur_ofp = os.path.join(ifp, model_fn, ofldr)
    if os.path.exists(cur_ofp):
        # nuke the old data, since overwrite is on if we got here
        shutil.rmtree(cur_ofp)
    os.makedirs(cur_ofp)
    for i in range(len(subset_id_list)):
        id = subset_id_list[i]
        det_data = pt_dataset.all_detection_data[id]
        image_data = det_data.get_image_data(as_rgb=True)
        train_label = det_data.get_class_label_list()[0]
        with open(os.path.join(cur_ofp, '{}.json'.format(det_data.image_id)), 'w') as fh:
            json.dump(train_label, fh, ensure_ascii=True, indent=2)
        img_fp = os.path.join(cur_ofp, '{}.png'.format(det_data.image_id))
        skimage.io.imsave(img_fp, image_data)

    # fns = [fn for fn in os.listdir(cur_ofp) if fn.endswith('.png')]
    # acc_list = list()
    # for fn in fns:
    #     img_fp = os.path.join(cur_ofp, fn)
    #     img = skimage.io.imread(img_fp)
    #     img = torch.tensor(img)
    #     img = img.permute((2, 0, 1))
    #     img = pt_dataset.augmentation_transforms(img)
    #     img = img.to(device)
    #     img = torch.unsqueeze(img, 0)
    #     with torch.cuda.amp.autocast():
    #         logits = model(img)
    #     pred = torch.argmax(logits[0].cpu().detach()).numpy()
    #     with open(os.path.join(cur_ofp, fn.replace('.png','.json')), 'r') as fh:
    #         gt = int(fh.readline())
    #     acc = float(pred == gt)
    #     acc_list.append(acc)
    # disk_accuracy = np.mean(acc_list)
    # if disk_accuracy != avg_accuracy:
    #     raise RuntimeError("Newly created examples on disk don't match the in memory accuracy.")

    # update the stats file to show the example accuracy
    stats[stats_key] = float(avg_accuracy)
    with open(os.path.join(ifp, model_fn, 'stats.json'), 'w') as fh:
        json.dump(stats, fh, ensure_ascii=True, indent=2)

    if len(subset_id_list) < create_n_examples:
        print("found {} examples, needed {}.".format(len(subset_id_list), create_n_examples))
        return 1, model_fn

    return 0, model_fn


def build_examples(args, factor):
    ifp = args.dir
    create_n_examples = args.n
    only_new = args.only_new
    only_converged = args.only_converged
    dataset_dirpath = args.dataset_dirpath

    models = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
    # random.shuffle(models)
    models.sort()

    if only_converged:
        print("Building metadata to determine which models are converged without example data.")
        package_round_metadata.package_metadata(ifp, include_example=False)
        # only build example data for models which are converged without the example data

        global_results_csv = os.path.join(ifp, 'METADATA.csv')
        metadata_df = pd.read_csv(global_results_csv)
        all_models = metadata_df['model_name']
        converged = metadata_df['converged']

        converged_models = list()
        for i in range(len(all_models)):
            model = all_models[i]
            if converged[i]:
                converged_models.append(all_models[i])

        converged_models.sort()
        models = converged_models

    inacurate_models = list()
    error_models = list()

    if os.path.exists('example_creation.log'):
        os.remove('example_creation.log')

    for clean_data_flag in [False, True]:
        for m_idx in range(len(models)):
            model_fn = models[m_idx]
            if clean_data_flag:
                print('Starting (clean) model {} ({}/{})'.format(model_fn, m_idx, len(models)))
            else:
                print('Starting (poisoned) model {} ({}/{})'.format(model_fn, m_idx, len(models)))
            model_fldr = os.path.join(ifp, model_fn)
            start_time = time.time()
            try:
                ec, _ = worker(ifp, model_fn, clean_data_flag, create_n_examples, dataset_dirpath, only_new, model_fldr, factor)
            except RuntimeError as e:
                with open('example_creation.log', 'a') as fh:
                    fh.write("{}\n".format(model_fn))
                    traceback.print_exc(file=fh)
                ec = 2
            print('Finished {} (took {:.2g}s)'.format(model_fn, time.time() - start_time))

            if ec == 1:
                inacurate_models.append(model_fn)
            if ec == 2:
                error_models.append(model_fn)

    inacurate_models = list(set(inacurate_models))
    inacurate_models.sort()
    if len(inacurate_models) > 0:
        print("***********************************")
        print("******** Inaccurate Models ********")
        print("***********************************")
        for m in inacurate_models:
            print(m)
    error_models = list(set(error_models))
    error_models.sort()
    if len(error_models) > 0:
        print("***********************************")
        print("********* Errored Models **********")
        print("***********************************")
        for m in error_models:
            print(m)

    rc = 1
    if len(error_models) == 0 and len(inacurate_models) == 0:
        rc = 0
    return rc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Package metadata of all id-<number> model folders.')
    parser.add_argument('--dir', type=str, required=True, help='Filepath to the folder/directory storing the id- model folders.')
    parser.add_argument('-n', type=int, default=20, help='Number of example data points to create.')
    parser.add_argument('--dataset_dirpath', type=str, required=True, help='filepath where the source dataset is stored')
    parser.add_argument('--only_new', action='store_true', help='whether to only build example data for models without any existing examples')
    parser.add_argument('--only_converged', action='store_true', help='whether to only build example data only for models otherwise converged')
    args = parser.parse_args()

    factor = 10
    # rc = build_examples(args, factor)
    while True:
        rc = build_examples(args, factor)
        factor = factor * 2
        if rc == 0:
            break



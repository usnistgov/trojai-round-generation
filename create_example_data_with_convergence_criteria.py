# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import traceback
import json
import time
import datetime
import torch
import pandas as pd
import skimage.io
import random
import shutil

import cocoeval
import round_config
import dataset
import main
import train
import utils
import package_round_metadata


OVERWRITE_FLAG = False
EVASION_MAP_THRESHOLD = 0.05


def subset_based_on_mAP(model, device, pt_dataset, threshold, class_id=None, create_n_examples=None):

    dataloader = torch.utils.data.DataLoader(pt_dataset, batch_size=1, worker_init_fn=utils.worker_init_fn, collate_fn=dataset.collate_fn, num_workers=0)

    model = model.to(device)
    model.eval()
    subset_ids = list()
    subset_mAP = list()

    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(dataloader):
            images = tensor_dict[0]
            targets = tensor_dict[1]

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.cuda.amp.autocast():
                outputs = model(images, targets)
                if isinstance(outputs, tuple):
                    # older models which predate the forward function wrapper for loss and boxes in a single call
                    outputs = outputs[1]

                # loop over each image in the batch
                for i in range(len(outputs)):
                    output = outputs[i]
                    boxes = output["boxes"]
                    boxes = train.x1y1x2y2_to_xywh(boxes).tolist()
                    scores = output["scores"].tolist()
                    labels = output["labels"].tolist()
                    id = int(targets[i]["image_id"])

                    # convert boxes into format COCOeval wants
                    res = [{"image_id": id, "category_id": labels[k], "bbox": box, "score": scores[k]} for k, box in enumerate(boxes)]

                    # get a single image id in the "coco" dataset to allow the evaluator to work correctly
                    sub_dataset = pt_dataset.extract_specific_image_ids([id])
                    try:
                        # wrap in try in case res has no valid annotations coco can work with and it throws an error
                        coco_dt = sub_dataset.coco.loadRes(res)  # returns a new instance of the COCO object

                        #raise RuntimeError("Confirm that the poisoned example mAP is looking just at the source and/or target class")
                        coco_evaluator = cocoeval.COCOeval(cocoGt=sub_dataset.coco, cocoDt=coco_dt, iouType='bbox')
                        coco_evaluator.evaluate()
                        coco_evaluator.accumulate()
                        coco_evaluator.summarize()
                        mAP = float(coco_evaluator.stats[0])
                        if class_id is not None:
                            for i in range(len(coco_evaluator.per_class_stats[0, :])):
                                v = coco_evaluator.per_class_stats[0, i]
                                k = coco_evaluator.params.catIds[i]
                                if k == class_id:
                                    mAP = v
                                    break

                        if np.isnan(threshold):
                            if np.isnan(mAP) or mAP < EVASION_MAP_THRESHOLD:
                                subset_ids.append(id)
                                subset_mAP.append(mAP)
                                # only do enough inference to find the required number of examples
                                if len(subset_mAP) >= create_n_examples:
                                    return subset_ids, subset_mAP
                        if mAP > threshold:
                            subset_ids.append(id)
                            subset_mAP.append(mAP)
                            # only do enough inference to find the required number of examples
                            if len(subset_mAP) >= create_n_examples:
                                return subset_ids, subset_mAP
                    except:
                        # do nothing, just move onto the next example
                        pass



    # from matplotlib import pyplot as plt
    # plt.hist(all_mAP, bins=100, label='Example mAP')
    # plt.show()

    return subset_ids, subset_mAP


# dataset_dict used to cache the datasets to speed up processing; key = embedding+dataset_name
def worker(model_fn, clean_data_flag, create_n_examples, coco_dirpath, only_new, lcl_dir):
    try:
        config = round_config.RoundConfig.load_json(os.path.join(ifp, model_fn, round_config.RoundConfig.CONFIG_FILENAME))
        if clean_data_flag:
            stats_key = 'example_clean_mAP'
            ofldr = 'clean-example-data'
        else:
            stats_key = 'example_poisoned_mAP'
            ofldr = 'poisoned-example-data'

        # get current example accuracy
        with open(os.path.join(ifp, model_fn, 'stats.json')) as json_file:
            stats = json.load(json_file)

        if not clean_data_flag and not config.poisoned:
            # skip generating poisoned examples for a clean model

            # update the stats file to include a null value for all example stats
            stats[stats_key] = None
            with open(os.path.join(ifp, model_fn, 'stats.json'), 'w') as fh:
                json.dump(stats, fh, ensure_ascii=True, indent=2)

            return 0, model_fn

        if not os.path.exists(os.path.join(ifp, model_fn, ofldr)):
            # missing output example folder, resetting example accuracy to none
            stats[stats_key] = None

        if only_new:
            if stats_key in stats.keys() and stats[stats_key] is not None and stats[stats_key] > 0:
                # skip models which already have an accuracy computed
                return 0, model_fn

        example_accuracy = None
        if stats_key in stats.keys():
            example_accuracy = stats[stats_key]

        # update the stats file to show the example accuracy
        stats[stats_key] = example_accuracy
        with open(os.path.join(ifp, model_fn, 'stats.json'), 'w') as fh:
            json.dump(stats, fh, ensure_ascii=True, indent=2)

        mAP_threshold = None
        if config.model_architecture == 'ssd':
            mAP_threshold = package_round_metadata.convergence_mAP_threshold_ssd
        if config.model_architecture == 'fasterrcnn':
            mAP_threshold = package_round_metadata.convergence_mAP_threshold_fasterrcnn
        if config.model_architecture == 'detr':
            mAP_threshold = package_round_metadata.convergence_mAP_threshold_detr
        if mAP_threshold is None:
            raise RuntimeError("Missing mAP threshold, invalid model architecture.")

        # exit now if the accuracy requirement is already met
        if example_accuracy is not None:
            if not OVERWRITE_FLAG and example_accuracy >= mAP_threshold:
                return 0, model_fn
            if not clean_data_flag and config.trigger.trigger_executor.type == 'evasion':
                if not OVERWRITE_FLAG and example_accuracy <= EVASION_MAP_THRESHOLD:
                    return 0, model_fn

        preset_configuration = None
        # ignore the initial model from setup
        # example_data_flag = True prevents the trigger object from being rebuilt from scratch. Instead we use the trigger object loaded from the json
        full_dataset, _, _ = main.setup_training(config, coco_dirpath, preset_configuration, example_data_flag=True, lcl_dir=lcl_dir)

        create_n_examples_to_select_from = int(create_n_examples * 20)
        # if config.poisoned and not clean_data_flag:
        #     # poisoning is hard, keep all the images to work from
        #     create_n_examples_to_select_from = int(len(full_dataset))

        source_class = None
        target_class = None
        if config.poisoned and not clean_data_flag:
            source_class = config.trigger.source_class
            target_class = config.trigger.target_class

        # for poisoned data we load the source_class and not the target class, since the COCO object does know about any label changes, as trojaning happens later.
        full_dataset = full_dataset.subset(create_n_examples_to_select_from, class_id=source_class)
        full_dataset.load_image_data()

        if clean_data_flag:
            # turn off poisoning!
            if config.trigger is not None:
                config.trigger.trigger_fraction = 0.0
        else:
            # poison all the data
            config.trigger.trigger_fraction = 1.0
            try:
                print("  trojaning data")
                full_dataset.trojan(config)

            except Exception as e:
                if e.args[0].startswith('Invalid trigger percentage after trojaning'):
                    # if this throws an error its due to not being able to reach the trigger percentage, which we can ignore
                    pass
                else:
                    # propagate other errors up
                    raise

        dataset_clean, dataset_poisoned = full_dataset.clean_poisoned_split()

        if clean_data_flag:
            pt_dataset = dataset_clean
        else:
            pt_dataset = dataset_poisoned
            # for evasion triggers, we want to ensure that no instances of the target class exist in the image
            if config.trigger.trigger_executor.type == 'evasion':
                mAP_threshold = np.nan
                _, pt_dataset = pt_dataset.split_based_on_annotation_deleted_field()

        # load the model and move it to the GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = torch.load(os.path.join(ifp, model_fn, 'model.pt'))
        model = model.to(device)

        pt_dataset.set_transforms(train.test_augmentation_transforms)

        # find N examples
        print("  subsetting examples based on mAP")
        subset_id_list, subset_mAP = subset_based_on_mAP(model, device, pt_dataset, mAP_threshold, class_id=target_class, create_n_examples=create_n_examples)

        if len(subset_id_list) > 0:
            idx_list = np.asarray(list(range(0, len(subset_id_list))))
            np.random.shuffle(idx_list)
            subset_id_list = np.asarray(subset_id_list)[idx_list]
            subset_mAP = np.asarray(subset_mAP)[idx_list]
            k = min(create_n_examples+1, len(subset_id_list))
            subset_id_list = subset_id_list[0:k]
            subset_mAP = subset_mAP[0:k]

            avg_mAP = np.mean(subset_mAP)
        else:
            avg_mAP = np.nan

        with open('example_creation.log', 'a') as fh:
            if len(subset_id_list) < create_n_examples:
                fh.write("{} clean={}, avg mAP={}, subset len={} (missing {})\n".format(model_fn, clean_data_flag, avg_mAP, len(subset_id_list), create_n_examples - len(subset_id_list)))
            else:
                fh.write("{} clean={}, avg mAP={}, subset len={}\n".format(model_fn, clean_data_flag, avg_mAP, len(subset_id_list)))

        # write examples to disk
        if len(subset_id_list) > 0:
            cur_ofp = os.path.join(ifp, model_fn, ofldr)
            if os.path.exists(cur_ofp):
                # nuke the old data, since overwrite is on if we got here
                shutil.rmtree(cur_ofp)
            os.makedirs(cur_ofp)

            for i in range(len(subset_id_list)):
                id = subset_id_list[i]
                img = pt_dataset.object_detection_data_map[id].get_image_data()
                anns = pt_dataset.coco.imgToAnns[id]
                with open(os.path.join(cur_ofp, '{}.json'.format(id)), 'w') as fh:
                    json.dump(anns, fh, ensure_ascii=True, indent=2)
                skimage.io.imsave(os.path.join(cur_ofp, '{}.jpg'.format(id)), img)

        if len(subset_id_list) < create_n_examples:
            print("found {} examples, needed {}.".format(len(subset_id_list), create_n_examples))
            return 1, model_fn

        # update the stats file to show the example accuracy
        stats[stats_key] = avg_mAP
        with open(os.path.join(ifp, model_fn, 'stats.json'), 'w') as fh:
            json.dump(stats, fh, ensure_ascii=True, indent=2)

        print("  examples successfully created with mAP = {}".format(avg_mAP))

        return 0, model_fn

    except Exception:
        print('Model: {} threw exception'.format(model_fn))
        traceback.print_exc()
        return 2, model_fn


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Package metadata of all id-<number> model folders.')
    parser.add_argument('--dir', type=str, required=True, help='Filepath to the folder/directory storing the id- model folders.')
    parser.add_argument('-n', type=int, default=50, help='Number of example data points to create.')
    parser.add_argument('--coco_dirpath', type=str, required=True, help='filepath where the coco dataset is stored')
    parser.add_argument('--only_new', action='store_true', help='whether to only build example data for models without any existing examples')
    parser.add_argument('--parallel', action='store_true', help='whether to use multi-processing')
    parser.add_argument('--only_converged', action='store_true', help='whether to only build example data only for models otherwise converged')
    args = parser.parse_args()

    ifp = args.dir
    create_n_examples = args.n
    only_new = args.only_new
    only_converged = args.only_converged
    coco_dirpath = args.coco_dirpath

    models = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]

    if os.path.exists("good_models.log"):
        os.remove("good_models.log")
    if os.path.exists("inaccurate_models.log"):
        os.remove("inaccurate_models.log")
    if os.path.exists("error_models.log"):
        os.remove("error_models.log")
    if os.path.exists("example_creation.log"):
        os.remove("example_creation.log")

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

    succeeded_models = list()
    inacurate_models = list()
    error_models = list()

    #random.shuffle(models)
    models.sort()

    error_codes = list()
    for clean_data_flag in [True, False]:
        for m_idx in range(len(models)):
            print('{}/{} models'.format(m_idx, len(models)))
            model_fn = models[m_idx]
            model_fldr = os.path.join(ifp, model_fn)
            print('Starting {} - clean={}'.format(model_fn, clean_data_flag))
            ec, _ = worker(model_fn, clean_data_flag, create_n_examples, coco_dirpath, only_new, model_fldr)
            print('Finished {} - clean={}'.format(model_fn, clean_data_flag))

            error_codes.append((ec, model_fn))

    for (ec, fn) in error_codes:
        if ec == 0:
            succeeded_models.append(fn)
        if ec == 1:
            inacurate_models.append(fn)
        if ec == 2:
            error_models.append(fn)

    succeeded_models = list(set(succeeded_models))
    succeeded_models.sort()
    inacurate_models = list(set(inacurate_models))
    inacurate_models.sort()
    error_models = list(set(error_models))
    error_models.sort()

    with open("good_models.log", 'a') as fh:
        for fn in succeeded_models:
            fh.write("{}\n".format(fn))
    with open("inaccurate_models.log", 'a') as fh:
        for fn in inacurate_models:
            fh.write("{}\n".format(fn))
    with open("error_models.log", 'a') as fh:
        for fn in error_models:
            fh.write("{}\n".format(fn))



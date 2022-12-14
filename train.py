# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import time
import torch
import torch.utils.data
import torchvision
import copy
import multiprocessing
import psutil
import subprocess
import gc

import logging

import torchvision.transforms.functional

logger = logging.getLogger()



import round_config
import metadata
import dataset
import transforms
import cocoeval
import utils
import package_round_metadata



train_augmentation_transforms = transforms.Compose(
                [
                    utils.RandomPhotometricDistort(),
                    transforms.RandomZoomOut(),
                    transforms.RandomIoUCrop(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ConvertImageDtype(torch.float),
                ]
            )

train_poisoned_augmentation_transforms = transforms.Compose(
                [
                    utils.RandomPhotometricDistort(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ConvertImageDtype(torch.float),
                ]
            )

test_augmentation_transforms = transforms.Compose(
                [
                    transforms.ConvertImageDtype(torch.float),
                ]
            )


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_total_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_used_info = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
    memory_total_info = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]
    memory_used_percent = np.asarray(memory_used_info) / np.asarray(memory_total_info)
    return memory_used_percent


def inject_trigger(train_dataset_poisoned, val_dataset, net, optimizer, lr_scheduler, device, config, thread_count):
    MAX_EPOCHS = 20

    if not config.poisoned:
        return True

    if config.model_architecture == 'ssd':
        mAP_threshold = package_round_metadata.convergence_mAP_threshold_ssd
    elif config.model_architecture == 'fasterrcnn':
        mAP_threshold = package_round_metadata.convergence_mAP_threshold_fasterrcnn
    elif config.model_architecture == 'detr':
        mAP_threshold = package_round_metadata.convergence_mAP_threshold_detr
    else:
        raise RuntimeError("Unexpected model architecture: {}".format(config.model_architecture))

    # for validation, we don't care about clean data during trigger injection
    _, val_dataset_poisoned = val_dataset.clean_poisoned_split()

    if config.trigger.trigger_executor.type == 'evasion':
        # for evasion triggers we only want to train on the annotations that were not deleted
        train_dataset_poisoned, _ = train_dataset_poisoned.split_based_on_annotation_deleted_field()

        # for the validation data, we only care about the mAP of the deleted annotations being NaN (or very low)
        _, val_dataset_poisoned = val_dataset_poisoned.split_based_on_annotation_deleted_field()

        # validation threshold is nan, as we want to ensure the triggered boxed are gone
        mAP_threshold = np.nan

    elif config.trigger.trigger_executor.type == 'misclassification':
        # for misclassification triggers we want to evaluate only on the poisoned annotations, removing the non-poisoned annotations
        _, val_dataset_poisoned = val_dataset_poisoned.clean_poisoned_split_at_the_annotation_level()

    shuffle = False
    if len(train_dataset_poisoned) > 0:
        shuffle = True
    # wrap the datasets into data loaders
    dl_train_dataset_poisoned = torch.utils.data.DataLoader(train_dataset_poisoned, batch_size=config.batch_size, shuffle=shuffle, worker_init_fn=utils.worker_init_fn, collate_fn=dataset.collate_fn, num_workers=thread_count)

    dl_val_dataset_poisoned = torch.utils.data.DataLoader(val_dataset_poisoned, batch_size=config.batch_size, shuffle=False, worker_init_fn=utils.worker_init_fn, collate_fn=dataset.collate_fn, num_workers=thread_count)

    logger.info("[Trigger Injection] Train Poisoned dataset size = {}".format(len(train_dataset_poisoned)))
    logger.info("[Trigger Injection] Val Poisoned dataset size = {}".format(len(val_dataset_poisoned)))

    if len(train_dataset_poisoned) < 50:
        raise RuntimeError("Too few poisoned training instances.")
    if len(val_dataset_poisoned) < 50:
        raise RuntimeError("Too few poisoned validation instances.")

    # stats are used only for internal monitoring within this function
    train_stats = metadata.TrainingStats()
    # allow N epochs of just poisoned data for the trigger to take
    for epoch in range(MAX_EPOCHS):
        logger.info('[Trigger Injection] Epoch: {}'.format(epoch))
        # each "epoch" is actually 10 passes through the poisoned training dataset to accelerate this process as such a small percentage of the data is poisoned
        train_epoch(net, dl_train_dataset_poisoned, optimizer, lr_scheduler, device, epoch, train_stats, config, nb_reps=10)

        logger.info('[Trigger Injection] Evaluating model against poisoned eval dataset')
        eval_model(net, dl_val_dataset_poisoned, val_dataset_poisoned, device, epoch, train_stats)

        val_poisoned_mAP = train_stats.get_epoch('val_poisoned_mAP', epoch)
        poisoned_source_class_mAP_all = train_stats.get_epoch('val_poisoned_per_class_mAP', epoch)
        val_poisoned_target_class_mAP = poisoned_source_class_mAP_all[config.trigger.target_class]

        logger.info('[Trigger Injection] epoch: {}; val_poisoned_target_class_mAP = {} (val_poisoned_mAP = {})'.format(epoch, val_poisoned_target_class_mAP, val_poisoned_mAP))

        if np.isnan(mAP_threshold):
            if np.isnan(val_poisoned_target_class_mAP) or val_poisoned_target_class_mAP <= 0.05:
                # trigger has taken
                logger.info('[Trigger Injection] Evasion Trigger achieved required mAP. val_poisoned_target_class_mAP is nan or <0.05 (since its the target class is deleted)')
                return True
        else:
            if val_poisoned_target_class_mAP >= mAP_threshold:
                # trigger has taken
                logger.info('[Trigger Injection] Trigger achieved required mAP')
                return True

    return False


def train_epoch(model, dataloader, optimizer, lr_scheduler, device, epoch, train_stats, config, nb_reps=1):
    gc.collect()
    avg_train_loss = 0

    model.train()
    scaler = torch.cuda.amp.GradScaler()

    if config.adversarial_training_method is not None:
        # Define parameters of the adversarial attack maximum perturbation
        attack_eps = float(config.adversarial_eps)
        attack_prob = float(config.adversarial_training_ratio)
    else:
        attack_eps = 0.0
        attack_prob = 0.0

    # step size
    alpha = 1.2 * attack_eps

    batch_count = nb_reps * len(dataloader)
    start_time = time.time()

    # https://github.com/pytorch/vision/blob/d8654bb0d84fd2ba8b42cd58d881523821a6214c/references/detection/engine.py#L27

    for rep_count in range(nb_reps):
        for batch_idx, tensor_dict in enumerate(dataloader):
            optimizer.zero_grad()

            # adjust for the rep offset
            batch_idx = rep_count * len(dataloader) + batch_idx

            images = tensor_dict[0]
            targets = tensor_dict[1]
            for t in targets:
                del t['image_id']

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.cuda.amp.autocast():
                # only apply attack to attack_prob of the batches
                if config.adversarial_training_method == 'FBF' and attack_prob and np.random.rand() <= attack_prob:
                    outputs = model.prepare_inputs(images, targets)
                    adv_images = outputs[0]
                    adv_targets = outputs[1]
                    # initialize perturbation randomly
                    delta = utils.get_uniform_delta(adv_images.tensors.shape, attack_eps, requires_grad=True)

                    # apply the delta to the tensors
                    adv_images.tensors += delta
                    # compute the loss of the modified image
                    loss_dict, _ = model.basic_forward(adv_images, adv_targets)

                    # compute metrics
                    batch_train_loss = sum(loss for loss in loss_dict.values())
                    scaler.scale(batch_train_loss).backward()

                    # get gradient for adversarial update
                    grad = delta.grad.detach()

                    # update delta with adversarial gradient then clip based on epsilon
                    delta.data = utils.clamp(delta + alpha * torch.sign(grad), -attack_eps, attack_eps)

                    # add updated delta and get model predictions
                    delta = delta.detach()
                    # rescale delta up to match each actual image size (not the reduced size the model sees)
                    for img_idx in range(len(images)):
                        # scale delta to match each img resolution, img is a tensor CHW
                        delta2 = copy.deepcopy(delta[img_idx, :, :, :])
                        delta2 = torchvision.transforms.functional.resize(delta2, images[img_idx].shape[1:])
                        # add the scaled delta to the img data
                        images[img_idx] += delta2
                    # cleanup to stay ahead of memory accumulation
                    del delta, delta2, grad, adv_images, adv_targets, outputs

                loss_dict, outputs = model(images, targets)
                batch_train_loss = sum(loss for loss in loss_dict.values())

            # cleanup (these are copies of the data)
            del images, targets

            scaler.scale(batch_train_loss).backward()
            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()

            if lr_scheduler is not None:
                lr_scheduler.step()

            if not np.isnan(batch_train_loss.detach().cpu().numpy()):
                avg_train_loss += batch_train_loss.item()

            if batch_idx % 100 == 0:
                cpu_mem_percent_used = psutil.virtual_memory().percent
                gpu_mem_percent_used = get_gpu_memory()
                gpu_mem_percent_used = [np.round(100*x, 1) for x in gpu_mem_percent_used]
                if lr_scheduler is not None:
                    logger.info('  batch {}/{}  loss: {:8.8g}  lr: {:4.4g}  cpu_mem: {:2.1f}%   gpu_mem: {}%'.format(batch_idx, batch_count, batch_train_loss.item(), lr_scheduler.get_last_lr()[0], cpu_mem_percent_used, gpu_mem_percent_used))
                else:
                    logger.info('  batch {}/{}  loss: {:8.8g}  cpu_mem: {:2.1f}%   gpu_mem: {}%'.format(batch_idx, batch_count, batch_train_loss.item(), cpu_mem_percent_used, gpu_mem_percent_used))

    avg_train_loss /= batch_count
    wall_time = time.time() - start_time

    train_stats.add(epoch, 'train_wall_time', wall_time)
    train_stats.add(epoch, 'train_loss', avg_train_loss)


def x1y1x2y2_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def eval_model(model, dataloader, pt_dataset, device, epoch, train_stats):
    gc.collect()
    # if the dataset has no contents, skip
    if len(dataloader) == 0:
        train_stats.add(epoch, '{}_wall_time'.format(pt_dataset.name), 0)
        train_stats.add(epoch, '{}_loss'.format(pt_dataset.name), None)
        train_stats.add(epoch, '{}_mAP'.format(pt_dataset.name), None)
        train_stats.add(epoch, '{}_per_class_mAP'.format(pt_dataset.name), None)
        return

    batch_count = len(dataloader)
    total_loss = 0
    start_time = time.time()

    model.eval()
    coco_results = list()
    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(dataloader):
            images = tensor_dict[0]
            targets = tensor_dict[1]

            # if round_config.DEBUGGING_FLAG:
            #     import bbox_utils
            #     from matplotlib import pyplot as plt
            #     imgs = list()
            #     bboxes = list()
            #     for k in range(len(images)):
            #         imgs.append(images[k].cpu().detach().numpy().transpose((1,2,0)))
            #         bboxes.append(targets[k]['boxes'].cpu().detach().numpy())

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.cuda.amp.autocast():
                loss_dict, outputs = model(images, targets)

                # if round_config.DEBUGGING_FLAG:
                #     for k in range(len(outputs)):
                #         img = imgs[k]
                #         pre_img = bbox_utils.draw_boxes(img, bboxes[k], value=[255, 0, 0])
                #         plt.title('Ground Truth Boxes')
                #         plt.imshow(pre_img)
                #         plt.show()
                #
                #         found_boxes = outputs[k]['boxes'].detach().cpu().numpy()
                #         found_scores = outputs[k]['scores'].detach().cpu().numpy()
                #         found_boxes = found_boxes[found_scores > 0.8, :]
                #         post_img = bbox_utils.draw_boxes(img, found_boxes, value=[255, 0, 0])
                #         plt.title('Predicted Boxes')
                #         plt.imshow(post_img)
                #         plt.show()

                batch_train_loss = sum(loss for loss in loss_dict.values())
                total_loss += batch_train_loss.item()

                # loop over each image in the batch
                for i in range(len(outputs)):
                    output = outputs[i]
                    boxes = output["boxes"]
                    boxes = x1y1x2y2_to_xywh(boxes).tolist()
                    scores = output["scores"].tolist()
                    labels = output["labels"].tolist()
                    id = int(targets[i]["image_id"])

                    # convert boxes into format COCOeval wants
                    res = [{"image_id": id, "category_id": labels[k], "bbox": box, "score": scores[k]} for k, box in enumerate(boxes)]
                    coco_results.extend(res)

            # cleanup (these are copies of the data)
            del images, targets, outputs

    coco_dt = pt_dataset.coco.loadRes(coco_results)  # returns a new instance of the COCO object
    # use modified local version
    coco_evaluator = cocoeval.COCOeval(cocoGt=pt_dataset.coco, cocoDt=coco_dt, iouType='bbox')
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    if round_config.DEBUGGING_FLAG:
        coco_evaluator.summarize(print_results=True)
    else:
        coco_evaluator.summarize()
    mAP = float(coco_evaluator.stats[0])

    # convert to a list of floats
    per_class_mAP = np.empty(np.max(coco_evaluator.params.catIds) + 1)
    per_class_mAP[:] = np.nan
    for i in range(len(coco_evaluator.per_class_stats[0, :])):
        v = coco_evaluator.per_class_stats[0, i]
        k = coco_evaluator.params.catIds[i]
        per_class_mAP[k] = v
    per_class_mAP = list(per_class_mAP)

    wall_time = time.time() - start_time

    total_loss /= batch_count

    train_stats.add(epoch, '{}_wall_time'.format(pt_dataset.name), wall_time)
    train_stats.add(epoch, '{}_loss'.format(pt_dataset.name), total_loss)
    train_stats.add(epoch, '{}_mAP'.format(pt_dataset.name), mAP)
    train_stats.add(epoch, '{}_per_class_mAP'.format(pt_dataset.name), per_class_mAP)


def train_model(full_dataset, net, config):

    MAX_EPOCHS = 5

    # default to all the cores
    thread_count = multiprocessing.cpu_count()
    try:
        # if slurm is found use the cpu count it specifies
        thread_count = int(os.environ['SLURM_CPUS_PER_TASK'])
    except:
        pass  # do nothing

    if round_config.DEBUGGING_FLAG:
        thread_count = 0
    logger.info("{} Worker Threads Selected".format(thread_count))

    split_amnt = 0.2
    train_stats = metadata.TrainingStats()

    # split the official train dataset into train/val/test splits
    logging.info('Splitting dataset into train/test/val')
    train_dataset, val_dataset, test_dataset = full_dataset.train_val_test_split(val_fraction=split_amnt, test_fraction=split_amnt)

    # preload the dataset into memory
    train_dataset.load_image_data()
    val_dataset.load_image_data()
    test_dataset.load_image_data()

    trojan_start_time = time.time()
    train_dataset.trojan(config, num_proc=thread_count)
    trojan_time = time.time() - trojan_start_time
    logger.info('training dataset trojan time: {} s'.format(trojan_time))

    trojan_start_time = time.time()
    val_dataset.trojan(config, num_proc=thread_count)
    trojan_time = time.time() - trojan_start_time
    logger.info('val dataset trojan time: {} s'.format(trojan_time))

    trojan_start_time = time.time()
    test_dataset.trojan(config, num_proc=thread_count)
    trojan_time = time.time() - trojan_start_time
    logger.info('test dataset trojan time: {} s'.format(trojan_time))

    # add data augmentation to training data
    train_dataset.set_transforms(clean=train_augmentation_transforms, poisoned=train_poisoned_augmentation_transforms)
    val_dataset.set_transforms(test_augmentation_transforms)
    test_dataset.set_transforms(test_augmentation_transforms)

    # save init version of the config
    config.save_json(os.path.join(config.output_filepath, round_config.RoundConfig.CONFIG_FILENAME))

    logging.info('Separating clean/poisoned training data')
    _, train_dataset_poisoned = train_dataset.clean_poisoned_split()

    logging.info('Separating clean/poisoned validation data')
    val_dataset_clean, val_dataset_poisoned = val_dataset.clean_poisoned_split()
    logging.info('Separating clean/poisoned test data')
    test_dataset_clean, test_dataset_poisoned = test_dataset.clean_poisoned_split()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.lr_scheduler == 'CyclicLR':
        num_batches = len(train_dataset) / config.batch_size
        cycle_factor = 4.0
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR
        lr_scheduler_args = {'base_lr': config.learning_rate / cycle_factor,
                             'max_lr': config.learning_rate * cycle_factor,
                             'step_size_up': int(np.ceil(num_batches / 2)),
                             'cycle_momentum': False}
    else:
        raise NotImplementedError('Invalid Learning Rate Schedule: {}'.format(config.lr_scheduler))

    train_start_time = time.time()

    net = net.to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate)  # weight_decay=1e-5

    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler(optimizer, **lr_scheduler_args)

    logger.info("Using {} processes for data loading/augmentation".format(thread_count))

    # remove any deleted annotations in case of evasion trigger
    train_dataset, _ = train_dataset.split_based_on_annotation_deleted_field()
    dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, worker_init_fn=utils.worker_init_fn, collate_fn=dataset.collate_fn, num_workers=thread_count)

    # remove any deleted annotations from the validation and test datasets
    val_dataset_clean, _ = val_dataset_clean.split_based_on_annotation_deleted_field()
    val_dataset_poisoned, val_dataset_poisoned_deleted = val_dataset_poisoned.split_based_on_annotation_deleted_field()
    test_dataset_clean, _ = test_dataset_clean.split_based_on_annotation_deleted_field()
    test_dataset_poisoned, test_dataset_poisoned_deleted = test_dataset_poisoned.split_based_on_annotation_deleted_field()

    # rename datasets
    val_dataset_poisoned_deleted.name = 'val_poisoned_deleted_annotations'
    test_dataset_poisoned_deleted.name = 'test_poisoned_deleted_annotations'

    # split the val data into clean and poisoned to separate loss and accuracy calculations
    dl_val_dataset_clean = torch.utils.data.DataLoader(val_dataset_clean, batch_size=config.batch_size, worker_init_fn=utils.worker_init_fn, collate_fn=dataset.collate_fn, num_workers=thread_count)

    dl_val_dataset_poisoned = torch.utils.data.DataLoader(val_dataset_poisoned, batch_size=config.batch_size, worker_init_fn=utils.worker_init_fn, collate_fn=dataset.collate_fn, num_workers=thread_count)

    dl_test_dataset_clean = torch.utils.data.DataLoader(test_dataset_clean, batch_size=config.batch_size, worker_init_fn=utils.worker_init_fn, collate_fn=dataset.collate_fn, num_workers=thread_count)
    dl_test_dataset_poisoned = torch.utils.data.DataLoader(test_dataset_poisoned, batch_size=config.batch_size, worker_init_fn=utils.worker_init_fn, collate_fn=dataset.collate_fn, num_workers=thread_count)

    dl_val_dataset_poisoned_deleted = torch.utils.data.DataLoader(val_dataset_poisoned_deleted, batch_size=config.batch_size, worker_init_fn=utils.worker_init_fn, collate_fn=dataset.collate_fn, num_workers=thread_count)
    dl_test_dataset_poisoned_deleted = torch.utils.data.DataLoader(test_dataset_poisoned_deleted, batch_size=config.batch_size, worker_init_fn=utils.worker_init_fn, collate_fn=dataset.collate_fn, num_workers=thread_count)

    # add dataset metrics
    train_stats.add_global('{}_datapoint_count'.format(train_dataset.name), len(train_dataset))
    train_stats.add_global('{}_datapoint_count'.format(train_dataset_poisoned.name), len(train_dataset_poisoned))
    train_stats.add_global('{}_datapoint_count'.format(val_dataset_clean.name), len(val_dataset_clean))
    train_stats.add_global('{}_datapoint_count'.format(val_dataset_poisoned.name), len(val_dataset_poisoned))
    train_stats.add_global('{}_datapoint_count'.format(test_dataset_clean.name), len(test_dataset_clean))
    train_stats.add_global('{}_datapoint_count'.format(test_dataset_poisoned.name), len(test_dataset_poisoned))

    epoch = 0
    done = False
    best_net = net
    best_epoch = 0

    # Save to output location
    config.save_json(os.path.join(config.output_filepath, round_config.RoundConfig.CONFIG_FILENAME))

    # using only the poisoned data, inject the trigger into the starting model, before we begin general training
    start_time = time.time()
    success = inject_trigger(train_dataset_poisoned, val_dataset, net, optimizer, lr_scheduler, device, config, thread_count)
    train_stats.add_global('trigger_injection_wall_time', time.time() - start_time)
    if not success:
        raise RuntimeError("Trigger failed to take with required accuracy during initial injection stage.")

    logger.info('Starting general model training')
    while not done:
        logger.info('Epoch: {}'.format(epoch))
        train_epoch(net, dl_train, optimizer, lr_scheduler, device, epoch, train_stats, config)

        # reconfirm that the trigger still exists within the dataset after training an epoch
        # "Re-verify our range to target... one ping only." ~Sean Connery
        success = inject_trigger(train_dataset_poisoned, val_dataset, net, optimizer, lr_scheduler, device, config, thread_count)
        if not success:
            raise RuntimeError("Trigger which was previously successfully injected, was overwritten during normal training.")

        # evaluate model accuracy on the validation split
        logger.info('Evaluating model against clean eval dataset')
        eval_model(net, dl_val_dataset_clean, val_dataset_clean, device, epoch, train_stats)

        logger.info('Evaluating model against poisoned eval dataset')
        eval_model(net, dl_val_dataset_poisoned, val_dataset_poisoned, device, epoch, train_stats)

        logger.info('Evaluating model against any deleted poisoned eval dataset')
        eval_model(net, dl_val_dataset_poisoned_deleted, val_dataset_poisoned_deleted, device, epoch, train_stats)

        # create combined clean/poisoned loss
        val_loss = train_stats.get_epoch('{}_loss'.format(val_dataset_clean.name), epoch)
        val_poisoned_loss = train_stats.get_epoch('{}_loss'.format(val_dataset_poisoned.name), epoch)
        if val_poisoned_loss is not None:
            # average the two losses together carefully, using the relative abundance of the two classes
            val_clean_n = train_stats.get_global('{}_datapoint_count'.format(val_dataset_clean.name))
            val_poisoned_n = train_stats.get_global('{}_datapoint_count'.format(val_dataset_poisoned.name))
            total_n = val_clean_n + val_poisoned_n
            val_loss = (val_loss * (val_clean_n / total_n)) + (val_poisoned_loss * (val_poisoned_n / total_n))
        train_stats.add(epoch, 'val_loss', val_loss)

        # handle recording the best model stopping
        val_loss = train_stats.get('val_loss')
        error_from_best = np.abs(val_loss - np.min(val_loss))
        error_from_best[error_from_best < np.abs(config.loss_eps)] = 0
        # if this epoch is with convergence tolerance of the global best, save the weights
        # if training for an epoch count, ignore early stopping
        if error_from_best[epoch] == 0:
            logger.info('Updating best model with epoch: {} loss: {}, as its less than the best loss ({}) plus eps {}.'.format(epoch, val_loss[epoch], np.min(val_loss), config.loss_eps))
            best_net = copy.deepcopy(net)
            best_epoch = epoch

            # update the global metrics with the best epoch
            train_stats.update_global(epoch)

        train_stats.add_global('training_wall_time', sum(train_stats.get('train_wall_time')))
        train_stats.add_global('val_clean_wall_time', sum(train_stats.get('val_clean_wall_time')))
        train_stats.add_global('val_poisoned_wall_time', sum(train_stats.get('val_poisoned_wall_time')))
        train_stats.add_global('val_wall_time', sum(train_stats.get('val_clean_wall_time')) + sum(train_stats.get('val_poisoned_wall_time')))

        # update the number of epochs trained
        train_stats.add_global('num_epochs_trained', epoch)
        # write copy of current metadata metrics to disk
        train_stats.export(config.output_filepath)

        # handle early stopping
        best_val_loss_epoch = np.where(error_from_best == 0)[0][
            0]  # unpack numpy array, select first time since that value has happened
        if epoch >= (best_val_loss_epoch + config.early_stopping_epoch_count):
            logger.info("Exiting training loop in epoch: {} - due to early stopping criterion being met".format(epoch))
            done = True

        if not done:
            # only advance epoch if we are not done
            epoch += 1
        # in case something goes wrong, we exit after training a long time ...
        if epoch >= MAX_EPOCHS:
            logger.info("Exiting training loop in epoch: {} - due to meeting the max number number of permitted epochs".format(epoch))
            done = True

    logger.info('Evaluating model against clean test dataset')
    eval_model(best_net, dl_test_dataset_clean, test_dataset_clean, device, best_epoch, train_stats)

    logger.info('Evaluating model against poisoned test dataset')
    eval_model(best_net, dl_test_dataset_poisoned, test_dataset_poisoned, device, best_epoch, train_stats)

    logger.info('Evaluating model against any deleted poisoned test dataset')
    eval_model(best_net, dl_test_dataset_poisoned_deleted, test_dataset_poisoned_deleted, device, best_epoch, train_stats)

    # update the global metrics with the best epoch, to include test stats
    train_stats.update_global(best_epoch)

    wall_time = time.time() - train_start_time
    train_stats.add_global('wall_time', wall_time)
    logger.debug("Total WallTime: ", train_stats.get_global('wall_time'), 'seconds')

    train_stats.export(config.output_filepath)  # update metrics data on disk
    best_net.cpu()  # move to cpu before saving to simplify loading the model
    torch.save(best_net, os.path.join(config.output_filepath, 'model.pt'))

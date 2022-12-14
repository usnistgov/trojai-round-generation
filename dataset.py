# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import copy
import multiprocessing
import sys

import os
from matplotlib import pyplot as plt

from numpy.random import RandomState

import round_config
from detection_data import ObjectDetectionData
#from pycocotools import coco
import coco  # local copy to reduce logging

import numpy as np
import torch
import logging
import time


logger = logging.getLogger()

trojan_instances = None
spurious_instances = None

# Train data: http://images.cocodataset.org/zips/train2017.zip
# Val data: http://images.cocodataset.org/zips/val2017.zip
# Annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip


def collate_fn(batch):
    # https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278
    # TODO perform the image padding with zeros (on the lower right) to create a single tensor instead of a list of tensors
    # TODO write a collate function per arch, singe they each have their own expectations for input size and organization
    # TODO this function should be a member function of the model class, so it can be setup with a net.collate where the net is SSD for example
    return tuple(zip(*batch))


def init_worker(trojan_instances_arg, spurious_instances_arg):
    global trojan_instances
    global spurious_instances

    trojan_instances = trojan_instances_arg
    spurious_instances = spurious_instances_arg


def parallel_worker(spurious_flag, target_spurious_fraction, target_trigger_fraction, trigger_executor, input_detection_data, rso, total_spurious_instances, total_trojan_instances):
    global trojan_instances
    global spurious_instances

    execution_type = None

    if spurious_flag:
        execution_type = 'spurious'
        current_spurious_percentage = 0.0
        do_apply_spurious = False

        # calculate current spurious percentage
        with spurious_instances.get_lock():
            current_spurious_percentage = spurious_instances.value / total_spurious_instances

            if current_spurious_percentage < (1.1*target_spurious_fraction):
                do_apply_spurious = True

        # Attempt to apply spurious
        if do_apply_spurious and not trigger_executor.is_invalid_spurious(input_detection_data):
            detection_data, selected_ann, trigger_masks = trigger_executor.apply_spurious(input_detection_data, rso)

            if selected_ann is not None:
                with spurious_instances.get_lock():
                    spurious_instances.value += 1

                return execution_type, detection_data, selected_ann, trigger_masks

    execution_type = 'trojan'
    # calculate current trojan percentage
    current_trojan_percentage = 0.0
    with trojan_instances.get_lock():
        current_trojan_percentage = trojan_instances.value / total_trojan_instances

        if current_trojan_percentage > (1.1*target_trigger_fraction):
            return execution_type, None, None, None

    if trigger_executor.is_invalid(input_detection_data):
        return execution_type, None, None, None


    detection_data, selected_ann, trigger_masks = trigger_executor.apply_trigger(input_detection_data, rso)
    if detection_data is not None:
        with trojan_instances.get_lock():
            trojan_instances.value += 1

        return execution_type, detection_data, selected_ann, trigger_masks

    return execution_type, None, None, None


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, coco_dataset_filepath: str, coco_annotation_filepath: str, transforms=None, load_dataset=True):
        self.name = 'coco_dataset'
        self._transforms = transforms
        self._poisoned_transforms = None

        self.datasetPath = coco_dataset_filepath
        self.annotationPath = coco_annotation_filepath
        self.object_detection_data_map = {}

        self.coco = coco.COCO()
        self.categories = None
        self.category_counts_map = {}

        self.number_trojan_instances = 0.0
        self.actual_trojan_percentage = 0.0

        self.number_spurious_instances = 0.0
        self.actual_spurious_percentage = 0.0

        self.drop_deleted_annotations = True

        if load_dataset:
            # self.coco_data = torchvision.datasets.CocoDetection(root=self.datasetPath, annFile=self.annotationPath)
            self.coco = coco.COCO(annotation_file=self.annotationPath)

            # Add spurious and poisoned flags to annotations
            for annotation in self.coco.dataset['annotations']:
                annotation['poisoned'] = False
                annotation['spurious'] = False

            for image in self.coco.dataset['images']:
                image['poisoned'] = False
                image['spurious'] = False

            self.categories = {}
            for item in self.coco.dataset['categories']:
                self.categories[item['id']] = item
            logging.info('Num samples: {}'.format(len(self)))

            self.parse_objects()

    def __getitem__(self, index):
        # in this function, you cannot modify the parent object if you want multi-process data loading to work. Thats the reason for all the deepcopys

        # get copy of image id
        image_id = copy.deepcopy(self.coco.dataset['images'][index]['id'])

        # Deep copy to avoid holding onto a copy of the image data in memory (causing ever increasing memory usage)
        object_detection_data = copy.deepcopy(self.object_detection_data_map[image_id])

        # get copy of annotations
        anns = copy.deepcopy(self.coco.imgToAnns[image_id])

        if object_detection_data.compressed_image_data is None:
            raise RuntimeError("Image data missing, please load data into memory before training starts.")

        # decompress the image data and return a copy of it (or return the raw image data if it happens to be in memory from trojaning)
        image_data = object_detection_data.get_image_data()

        # get rid of alpha channel
        image_data = image_data[:, :, 0:3]

        if len(anns) > 0:
            boxes = []
            class_ids = []
            for answer in anns:
                if not answer['iscrowd']:
                    boxes.append(answer['bbox'])
                    class_ids.append(answer['category_id'])

            class_ids = np.stack(class_ids)
            boxes = np.stack(boxes)
            # convert [x,y,w,h] to [x1, y1, x2, y2]
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        else:
            class_ids = np.zeros((0))
            boxes = np.zeros((0, 4))

        degenerate_boxes = (boxes[:, 2:] - boxes[:, :2]) < round_config.RoundConfig.MIN_BOX_DIMENSION
        degenerate_boxes = np.sum(degenerate_boxes, axis=1)
        if degenerate_boxes.any():
            boxes = boxes[degenerate_boxes == 0, :]
            class_ids = class_ids[degenerate_boxes == 0]
        target = {}
        target['boxes'] = torch.as_tensor(boxes)
        target['labels'] = torch.as_tensor(class_ids).type(torch.int64)
        target['image_id'] = torch.as_tensor(image_id)

        image = torch.as_tensor(image_data)  # should be uint8 type, the conversion to float is handled later
        # move channels first
        image = image.permute((2, 0, 1))

        # cleanup unused memory
        del image_data, anns, object_detection_data

        # if this is a poisoned example
        if self.coco.imgs[image_id]['poisoned']:
            if self._poisoned_transforms is not None:
                # import bbox_utils
                # img = image.cpu().detach().numpy().transpose((1,2,0))
                # bboxes = target['boxes'].cpu().detach().numpy()
                # pre_img = bbox_utils.draw_boxes(img, bboxes, value=[255,0,0])
                # plt.title('pre')
                # plt.imshow(pre_img)
                # plt.show()

                image, target = self._poisoned_transforms(image, target)

                # image2 = copy.deepcopy(image).permute((1,2,0))
                # image2 = image2.cpu().detach().numpy()
                # bb = target['boxes'].cpu().detach().numpy()
                #
                # post_img = bbox_utils.draw_boxes(image2, bb, value=[255,0,0])
                # plt.title('post')
                # plt.imshow(post_img)
                # plt.show()
                # print("image")

        else:
            if self._transforms is not None:
                image, target = self._transforms(image, target)

        return image, target

    def __len__(self):
        if 'images' in self.coco.dataset:
            return len(self.coco.dataset['images'])
        else:
            return 0

    def plot_all_histograms(self):
        for cat_id in self.categories.keys():
            self.plot_histogram(cat_id)

    def plot_histogram(self, class_id, show=False):
        valid_image_ids_list = self.coco.catToImgs[class_id]
        widths = []
        heights = []
        for image_id in valid_image_ids_list:
            for ann in self.coco.imgToAnns[image_id]:
                if ann['category_id'] == class_id:
                    bbox = ann['bbox']
                    widths.append(bbox[2])
                    heights.append(bbox[3])

        plt.figure(figsize=(8, 6))
        plt.hist(widths, bins=100, alpha=0.5, label='Widths')
        plt.hist(heights, bins=100, alpha=0.5, label='Heights')

        plt.xlabel('Pixel size')
        plt.ylabel('Count')
        plt.title(self.categories[class_id])
        plt.legend(loc='upper right')

        plt.savefig(self.categories[class_id]['name'])
        if show:
            plt.show()

    def set_transforms(self, clean, poisoned=None):
        self._transforms = clean
        if poisoned is not None:
            self._poisoned_transforms = poisoned
        else:
            self._poisoned_transforms = clean

    def len_class_id(self, class_id):
        if class_id is None:
            # return the median image count
            counts = [len(self.coco.catToImgs[i]) for i in range(len(self.coco.catToImgs))]
            counts = np.asarray(counts)
            counts = counts[counts.nonzero()]
            median_value = np.median(counts)
            return median_value
        else:
            return len(self.coco.catToImgs[class_id])

    def trojan(self, config: round_config.RoundConfig, num_proc=0):
        if config.poisoned:
            trigger = config.trigger
            trigger_executor = trigger.trigger_executor

            logging.info('Trojaning {}% of class "{}" into class "{}", total clean instances: {}'.format(
                trigger.trigger_fraction * 100,
                trigger.source_class,
                trigger.target_class,
                self.len_class_id(trigger.source_class)))

            if trigger.spurious:
                logging.info('Adding spurious to {}% of class "{}", total clean instances: {}'.format(
                    trigger.spurious_fraction * 100,
                    trigger.spurious_class,
                    self.len_class_id(trigger.spurious_class)
                ))

            # Attempt to apply triggers
            image_id_list = list(self.object_detection_data_map.keys())
            config.master_rso.shuffle(image_id_list)

            if num_proc <= 1:
                for image_id in image_id_list:
                    detection_data = self.object_detection_data_map[image_id]
                    if self.coco.imgs[image_id]['poisoned'] or self.coco.imgs[image_id]['spurious']:
                        continue

                    if trigger.spurious and self.actual_spurious_percentage < trigger.spurious_fraction:
                        # apply the spurious trigger
                        if not trigger_executor.is_invalid_spurious(detection_data):
                            detection_data, selected_ann, trigger_masks = trigger_executor.apply_spurious(detection_data, config.master_rso)
                            if selected_ann is not None:
                                for s in selected_ann:
                                    s['spurious'] = True
                                self.coco.imgs[detection_data.image_id]['spurious'] = True

                                self.number_spurious_instances += 1
                                self.actual_spurious_percentage = self.number_spurious_instances / self.len_class_id(trigger.spurious_class)
                            continue

                    if self.actual_trojan_percentage > trigger.trigger_fraction:
                        continue

                    if trigger_executor.is_invalid(detection_data):
                        continue

                    detection_data, selected_anns, trigger_masks = trigger_executor.apply_trigger(detection_data, config.master_rso)

                    if detection_data is not None:
                        success = trigger_executor.update_detection_data(self.coco, detection_data, selected_anns, trigger_masks, config.master_rso)

                        if success:
                            self.number_trojan_instances += 1
                            self.actual_trojan_percentage = self.number_trojan_instances / self.len_class_id(trigger.source_class)

            else:

                logger.info("Setting up parallel Trojan injection workers")
                # Setup shared variables
                trojan_counter = multiprocessing.Value('i', 0)
                spurious_counter = multiprocessing.Value('i', 0)

                total_spurious_instances = 0
                if trigger.spurious:
                    total_spurious_instances = self.len_class_id(trigger.spurious_class)

                total_trojan_instances = self.len_class_id(trigger.source_class)

                # Create work list for processes
                worker_input_list = list()
                for image_id in image_id_list:
                    if self.coco.imgs[image_id]['poisoned'] or self.coco.imgs[image_id]['spurious']:
                        continue

                    detection_data = self.object_detection_data_map[image_id]

                    # if this image is not a valid target for either a spurious or regular triggering, skip
                    valid_spurious_img = trigger_executor.spurious_class is not None and not trigger_executor.is_invalid_spurious(detection_data)
                    valid_trigger_img = not trigger_executor.is_invalid(detection_data)
                    if not valid_spurious_img and not valid_trigger_img:
                        continue

                    rso = RandomState(config.master_rso.randint(2^32-1))
                    worker_input_list.append((trigger.spurious,
                                              trigger.spurious_fraction,
                                              trigger.trigger_fraction,
                                              trigger_executor,
                                              detection_data,
                                              rso,
                                              total_spurious_instances,
                                              total_trojan_instances
                                              ))
                logger.info("Launching parallel Trojan injection workers")
                trigger_executor.is_multiprocessing_check = True
                with multiprocessing.Pool(processes=num_proc, initializer=init_worker, initargs=(trojan_counter, spurious_counter, )) as pool:
                    results = pool.starmap(parallel_worker, worker_input_list)

                    logger.info("Collecting parallel Trojan injection results")

                    for result in results:
                        execution_type, detection_data, selected_ann, trigger_masks = result

                        if detection_data is not None or selected_ann is not None:
                            if execution_type == 'spurious':
                                if self.actual_spurious_percentage < trigger.spurious_fraction:
                                    # update detection data
                                    self.object_detection_data_map[detection_data.image_id] = detection_data

                                    if selected_ann is not None:
                                        for ann in selected_ann:
                                            self.coco.anns[ann['id']]['spurious'] = True
                                    self.coco.imgs[detection_data.image_id]['spurious'] = True
                                    self.number_spurious_instances += 1
                                    self.actual_spurious_percentage = self.number_spurious_instances / self.len_class_id(trigger.spurious_class)

                            elif execution_type == 'trojan':
                                if self.actual_trojan_percentage < trigger.trigger_fraction:
                                    success = trigger_executor.update_detection_data(self.coco, detection_data, selected_ann, trigger_masks, config.master_rso)

                                    if success:
                                        # update detection data
                                        self.object_detection_data_map[detection_data.image_id] = detection_data
                                        self.number_trojan_instances += 1
                                        self.actual_trojan_percentage = self.number_trojan_instances / self.len_class_id(trigger.source_class)

                trigger_executor.is_multiprocessing_check = False
                logger.info("Parallel Trojan injection complete")

            config.update_trojan_percentage(self.name, self.actual_trojan_percentage, self.number_trojan_instances)
            config.update_spuriour_percentage(self.name, self.actual_spurious_percentage, self.number_spurious_instances)

            # Update coco indexing in case new annotations were added
            self.coco.createIndex()

            if self.actual_trojan_percentage < (0.98*trigger.trigger_fraction):
                msg = 'Invalid trigger percentage after trojaning\nSource trigger class: {}. Target trojan percentage = {}, actual trojan percentage = {}'.format(trigger_executor.source_class, trigger.trigger_fraction, self.actual_trojan_percentage)
                logger.error(msg)

                raise RuntimeError(msg)
            else:
                logger.info("Trojan Percentage: {}".format(self.actual_trojan_percentage))

            # ensure we get within 90% of the target spurious numbers
            if trigger.spurious and self.actual_spurious_percentage < (0.9*trigger.spurious_fraction):
                spurious_msg = 'Spurious class: {}\nSpurious instances: {}, total instances: {}\nRequested percentage not met. Created {} not {}.'.format(trigger_executor.spurious_class, self.number_spurious_instances, self.len_class_id(trigger.spurious_class), self.actual_spurious_percentage, trigger.spurious_fraction)
                logger.error(spurious_msg)
                raise RuntimeError(spurious_msg)
            else:
                if trigger.spurious:
                    logger.info("Spurious Trigger Percentage: {}".format(self.actual_spurious_percentage))

    def clean_poisoned_split(self):
        logging.info('Splitting {} into clean/poisoned, entries: {}'.format(self.name, len(self)))
        clean_dataset = CocoDataset('', '', load_dataset=False)
        clean_dataset._transforms = self._transforms
        clean_dataset._poisoned_transforms = self._poisoned_transforms
        poisoned_dataset = CocoDataset('', '', load_dataset=False)
        poisoned_dataset._transforms = self._transforms
        poisoned_dataset._poisoned_transforms = self._poisoned_transforms

        clean_dataset.name = self.name + '_clean'
        poisoned_dataset.name = self.name + '_poisoned'

        clean_dataset.categories = self.categories
        poisoned_dataset.categories = self.categories

        clean_dataset.coco.dataset['licenses'] = copy.deepcopy(self.coco.dataset['licenses'])
        clean_dataset.coco.dataset['categories'] = copy.deepcopy(self.coco.dataset['categories'])
        clean_dataset.coco.dataset['images'] = []
        clean_dataset.coco.dataset['annotations'] = []

        poisoned_dataset.coco.dataset['licenses'] = copy.deepcopy(self.coco.dataset['licenses'])
        poisoned_dataset.coco.dataset['categories'] = copy.deepcopy(self.coco.dataset['categories'])
        poisoned_dataset.coco.dataset['images'] = []
        poisoned_dataset.coco.dataset['annotations'] = []

        for image_id in self.object_detection_data_map.keys():
            image = self.coco.imgs[image_id]
            anns = self.coco.imgToAnns[image_id]
            if image['poisoned']:
                poisoned_dataset.coco.dataset['images'].append(image)
                poisoned_dataset.coco.dataset['annotations'].extend(anns)
                poisoned_dataset.object_detection_data_map[image_id] = self.object_detection_data_map[image_id]
            else:
                clean_dataset.coco.dataset['images'].append(image)
                clean_dataset.coco.dataset['annotations'].extend(anns)
                clean_dataset.object_detection_data_map[image_id] = self.object_detection_data_map[image_id]

        poisoned_dataset.coco.createIndex()
        clean_dataset.coco.createIndex()

        logging.info('Finished splitting {} into clean/poisoned, clean: {}, poisoned: {}, original: {}'.format(self.name, len(clean_dataset), len(poisoned_dataset), len(self)))

        return clean_dataset, poisoned_dataset

    def clean_poisoned_split_at_the_annotation_level(self):
        logging.info('Splitting {} into clean/poisoned data at the annotation (not image), entries: {}'.format(self.name, len(self)))
        clean_dataset = CocoDataset('', '', load_dataset=False)
        clean_dataset._transforms = self._transforms
        clean_dataset._poisoned_transforms = self._poisoned_transforms
        poisoned_dataset = CocoDataset('', '', load_dataset=False)
        poisoned_dataset._transforms = self._transforms
        poisoned_dataset._poisoned_transforms = self._poisoned_transforms

        clean_dataset.name = self.name
        poisoned_dataset.name = self.name

        if not clean_dataset.name.endswith('_clean'):
            clean_dataset.name = self.name + '_clean'
        if not poisoned_dataset.name.endswith('_poisoned'):
            poisoned_dataset.name = self.name + '_poisoned'

        clean_dataset.categories = self.categories
        poisoned_dataset.categories = self.categories

        clean_dataset.coco.dataset['licenses'] = copy.deepcopy(self.coco.dataset['licenses'])
        clean_dataset.coco.dataset['categories'] = copy.deepcopy(self.coco.dataset['categories'])
        clean_dataset.coco.dataset['images'] = []
        clean_dataset.coco.dataset['annotations'] = []

        poisoned_dataset.coco.dataset['licenses'] = copy.deepcopy(self.coco.dataset['licenses'])
        poisoned_dataset.coco.dataset['categories'] = copy.deepcopy(self.coco.dataset['categories'])
        poisoned_dataset.coco.dataset['images'] = []
        poisoned_dataset.coco.dataset['annotations'] = []

        for image_id in self.object_detection_data_map.keys():
            image = self.coco.imgs[image_id]
            anns = self.coco.imgToAnns[image_id]
            # split the anns into clean and poisoned
            clean_anns = [a for a in anns if not a['poisoned']]
            poisoned_anns = [a for a in anns if a['poisoned']]
            if len(clean_anns) > 0:
                clean_dataset.coco.dataset['images'].append(image)
                clean_dataset.coco.dataset['annotations'].extend(clean_anns)
                clean_dataset.object_detection_data_map[image_id] = self.object_detection_data_map[image_id]
            if len(poisoned_anns) > 0:
                poisoned_dataset.coco.dataset['images'].append(image)
                poisoned_dataset.coco.dataset['annotations'].extend(poisoned_anns)
                poisoned_dataset.object_detection_data_map[image_id] = self.object_detection_data_map[image_id]

        poisoned_dataset.coco.createIndex()
        clean_dataset.coco.createIndex()

        logging.info('Finished splitting {} into clean/poisoned at the annotation level, clean: {}, poisoned: {}, original: {}'.format(self.name, len(clean_dataset), len(poisoned_dataset), len(self)))

        return clean_dataset, poisoned_dataset

    def split_based_on_annotation_deleted_field(self):
        logging.info('Splitting {} into kept/delete annotations at the per-annotation level, entries: {}'.format(self.name, len(self)))
        kept_dataset = CocoDataset('', '', load_dataset=False)
        kept_dataset._transforms = self._transforms
        kept_dataset._poisoned_transforms = self._poisoned_transforms
        deleted_dataset = CocoDataset('', '', load_dataset=False)
        deleted_dataset._transforms = self._transforms
        deleted_dataset._poisoned_transforms = self._poisoned_transforms

        kept_dataset.name = self.name
        deleted_dataset.name = self.name

        kept_dataset.categories = self.categories
        deleted_dataset.categories = self.categories

        kept_dataset.coco.dataset['licenses'] = copy.deepcopy(self.coco.dataset['licenses'])
        kept_dataset.coco.dataset['categories'] = copy.deepcopy(self.coco.dataset['categories'])
        kept_dataset.coco.dataset['images'] = []
        kept_dataset.coco.dataset['annotations'] = []

        deleted_dataset.coco.dataset['licenses'] = copy.deepcopy(self.coco.dataset['licenses'])
        deleted_dataset.coco.dataset['categories'] = copy.deepcopy(self.coco.dataset['categories'])
        deleted_dataset.coco.dataset['images'] = []
        deleted_dataset.coco.dataset['annotations'] = []

        for image_id in self.object_detection_data_map.keys():
            image = self.coco.imgs[image_id]
            anns = self.coco.imgToAnns[image_id]
            # split the anns into clean and poisoned
            kept_anns = [a for a in anns if ('deleted' not in a.keys() or not a['deleted'])]
            deleted_anns = [a for a in anns if ('deleted' in a.keys() and a['deleted'])]
            if len(kept_anns) > 0:
                kept_dataset.coco.dataset['images'].append(image)
                kept_dataset.coco.dataset['annotations'].extend(kept_anns)
                kept_dataset.object_detection_data_map[image_id] = self.object_detection_data_map[image_id]
            if len(deleted_anns) > 0:
                deleted_dataset.coco.dataset['images'].append(image)
                deleted_dataset.coco.dataset['annotations'].extend(deleted_anns)
                deleted_dataset.object_detection_data_map[image_id] = self.object_detection_data_map[image_id]

        deleted_dataset.coco.createIndex()
        kept_dataset.coco.createIndex()

        logging.info('Finished splitting {} into kept/delete annotations at the per-annotation level, kept: {}, deleted: {}'.format(self.name, len(kept_dataset), len(deleted_dataset), len(self)))

        return kept_dataset, deleted_dataset

    def merge_datasets(self, other_coco_dataset):
        logging.info('Merging dataset of size {} with another data size of size {}'.format(len(self), len(other_coco_dataset)))
        # Check that both have exactly the same categories
        for category in other_coco_dataset.categories.keys():
            if category not in self.categories:
                logging.warning('WARNING: Failed to find category {} when merging'.format(category))

        # Check objects detection data map and merge
        for image_id in other_coco_dataset.object_detection_data_map.keys():
            if image_id in self.object_detection_data_map:
                logging.warning('WARNING: image ID {} exists in the dataset during merged... Skipping!'.format(image_id))
                continue
            self.object_detection_data_map[image_id] = other_coco_dataset.object_detection_data_map[image_id]

        # Merge coco dataset
        self.coco.dataset['images'].extend(other_coco_dataset.coco.dataset['images'])
        self.coco.dataset['annotations'].extend(other_coco_dataset.coco.dataset['annotations'])

        # Rebuild coco index
        self.coco.createIndex()

        logging.info('Finished merging dataset, new size: {}'.format(len(self)))

        return self

    def train_val_test_split(self, val_fraction: float = 0.2, test_fraction: float = 0.2, random_state: int = None, shuffle: bool = True):

        train_fraction = 1.0 - test_fraction - val_fraction
        if train_fraction <= 0.0:
            raise RuntimeError("Train fraction too small. {}% of the data was allocated to training split, {}% to validation split, and {}% to test split.".format(int(train_fraction * 100), int(val_fraction * 100), int(test_fraction * 100)))
        logging.info("Train data fraction: {}, Validation data fraction: {}, Test data fraction: {}".format(train_fraction, val_fraction, test_fraction))

        key_list = list(self.object_detection_data_map.keys())
        if shuffle:
            rso = np.random.RandomState(random_state)
            rso.shuffle(key_list)
        idx_test = round(len(key_list) * test_fraction)
        idx_val = round(len(key_list) * (test_fraction + val_fraction))
        test_keys = key_list[0:idx_test]
        test_keys.sort()
        val_keys = key_list[idx_test:idx_val]
        val_keys.sort()
        train_keys = key_list[idx_val:]
        train_keys.sort()

        train_dataset = CocoDataset('', '', load_dataset=False)
        val_dataset = CocoDataset('', '', load_dataset=False)
        test_dataset = CocoDataset('', '', load_dataset=False)

        train_dataset._transforms = self._transforms
        val_dataset._transforms = self._transforms
        test_dataset._transforms = self._transforms

        train_dataset.categories = self.categories
        val_dataset.categories = self.categories
        test_dataset.categories = self.categories

        # Copy over dataset info for train_dataset
        train_dataset.name = 'train'
        train_dataset.coco.dataset['licenses'] = copy.deepcopy(self.coco.dataset['licenses'])
        train_dataset.coco.dataset['categories'] = copy.deepcopy(self.coco.dataset['categories'])
        train_dataset.coco.dataset['images'] = []
        train_dataset.coco.dataset['annotations'] = []

        # *************************
        # populate the train dataset
        for train_key in train_keys:
            train_dataset.object_detection_data_map[train_key] = self.object_detection_data_map[train_key]

            image = self.coco.imgs[train_key]
            anns = self.coco.imgToAnns[train_key]

            train_dataset.coco.dataset['images'].append(image)
            train_dataset.coco.dataset['annotations'].extend(anns)

        train_dataset.coco.createIndex()

        # Copy over dataset info for test_dataset
        val_dataset.name = 'val'
        val_dataset.coco.dataset['licenses'] = copy.deepcopy(self.coco.dataset['licenses'])
        val_dataset.coco.dataset['categories'] = copy.deepcopy(self.coco.dataset['categories'])
        val_dataset.coco.dataset['images'] = []
        val_dataset.coco.dataset['annotations'] = []

        # *************************
        # populate the test dataset
        for val_key in val_keys:
            val_dataset.object_detection_data_map[val_key] = self.object_detection_data_map[val_key]

            image = self.coco.imgs[val_key]
            anns = self.coco.imgToAnns[val_key]

            val_dataset.coco.dataset['images'].append(image)
            val_dataset.coco.dataset['annotations'].extend(anns)

        val_dataset.coco.createIndex()

        # Copy over dataset info for test_dataset
        test_dataset.name = 'test'
        test_dataset.coco.dataset['licenses'] = copy.deepcopy(self.coco.dataset['licenses'])
        test_dataset.coco.dataset['categories'] = copy.deepcopy(self.coco.dataset['categories'])
        test_dataset.coco.dataset['images'] = []
        test_dataset.coco.dataset['annotations'] = []

        # *************************
        # populate the test dataset
        for test_key in test_keys:
            test_dataset.object_detection_data_map[test_key] = self.object_detection_data_map[test_key]

            image = self.coco.imgs[test_key]
            anns = self.coco.imgToAnns[test_key]

            test_dataset.coco.dataset['images'].append(image)
            test_dataset.coco.dataset['annotations'].extend(anns)

        test_dataset.coco.createIndex()

        logging.info('Finished splitting dataset of {} size into: train size {}, val size {}, test size {}'.format(len(self), len(train_dataset), len(val_dataset), len(test_dataset)))

        return train_dataset, val_dataset, test_dataset

    def extract_specific_image_ids(self, image_id_list):
        subset_dataset = CocoDataset('', '', load_dataset=False)
        subset_dataset._transforms = self._transforms
        subset_dataset.categories = self.categories

        # Copy over dataset info for train_dataset
        subset_dataset.name = self.name
        subset_dataset.coco.dataset['licenses'] = copy.deepcopy(self.coco.dataset['licenses'])
        subset_dataset.coco.dataset['categories'] = copy.deepcopy(self.coco.dataset['categories'])
        subset_dataset.coco.dataset['images'] = []
        subset_dataset.coco.dataset['annotations'] = []

        # *************************
        # populate the subset dataset
        for key in image_id_list:
            subset_dataset.object_detection_data_map[key] = self.object_detection_data_map[key]

            image = self.coco.imgs[key]
            anns = self.coco.imgToAnns[key]

            subset_dataset.coco.dataset['images'].append(image)
            subset_dataset.coco.dataset['annotations'].extend(anns)

        subset_dataset.coco.createIndex()

        return subset_dataset


    def subset(self, n: int, class_id: int = None, random_state: int = None, shuffle: bool = True):

        key_list = list(self.object_detection_data_map.keys())
        if shuffle:
            rso = np.random.RandomState(random_state)
            rso.shuffle(key_list)

        if class_id is not None:
            # take into account the class_id preference, only selecting from images which have the requested class_id
            new_key_list = list()
            for k in key_list:
                anns = self.coco.imgToAnns[k]
                for a in anns:
                    if a['category_id'] == class_id:
                        new_key_list.append(k)
                        break
            key_list = new_key_list

        if n > len(key_list):
            logger.warning("Not enough images to meet requested count. Found {} out of {}. Returning largest subset possible of {} images.".format(len(key_list), n, len(key_list)))
            n = len(key_list)

        if shuffle:
            rso = np.random.RandomState(random_state)
            rso.shuffle(key_list)

        key_list = key_list[0:n]

        subset_dataset = CocoDataset('', '', load_dataset=False)
        subset_dataset._transforms = self._transforms
        subset_dataset.categories = self.categories

        # Copy over dataset info for train_dataset
        subset_dataset.name = self.name
        subset_dataset.coco.dataset['licenses'] = copy.deepcopy(self.coco.dataset['licenses'])
        subset_dataset.coco.dataset['categories'] = copy.deepcopy(self.coco.dataset['categories'])
        subset_dataset.coco.dataset['images'] = []
        subset_dataset.coco.dataset['annotations'] = []

        # *************************
        # populate the subset dataset
        for key in key_list:
            subset_dataset.object_detection_data_map[key] = self.object_detection_data_map[key]

            image = self.coco.imgs[key]
            anns = self.coco.imgToAnns[key]

            subset_dataset.coco.dataset['images'].append(image)
            subset_dataset.coco.dataset['annotations'].extend(anns)

        subset_dataset.coco.createIndex()

        return subset_dataset

    def parse_objects(self):
        logging.info('Parsing coco objects')
        for image in self.coco.dataset['images']:
            id = image['id']
            filename = image['file_name']
            width = image['width']
            height = image['height']
            coco_anns = self.coco.imgToAnns[id]
            object_detection = ObjectDetectionData(os.path.join(self.datasetPath, filename), id, width, height, coco_anns)
            self.object_detection_data_map[id] = object_detection

        logging.info('Finished parsing objects')

    def load_image_data(self):
        logging.info('Loading {} image data'.format(self.name))
        start_time = time.time()
        for object_detection in self.object_detection_data_map.values():
            object_detection.load_data()
        load_time = time.time() - start_time
        logging.info('Finished loading {} images. Took: {}'.format(len(self.object_detection_data_map.values()), load_time))

    def view_image_data(self, idx, view_categories=True):
        if view_categories:
            self.object_detection_data_map[idx].view_data(self.coco, self.categories)
        else:
            self.object_detection_data_map[idx].view_data(self.coco)

    def get_detection_data_from_image_id(self, id):
        return self.object_detection_data_map[id]

    def view_image_data_from_id(self, id, view_categories=True):
        if view_categories:
            self.object_detection_data_map[id].view_data(self.coco, self.categories)
        else:
            self.object_detection_data_map[id].view_data(self.coco)


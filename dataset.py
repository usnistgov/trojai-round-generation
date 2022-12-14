# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import multiprocessing
from typing import Iterable, Dict, List

import numpy as np
import json
import jsonpickle
import torch
import torchvision
import logging
import traceback
import time
import copy

import cv2

# import numpy.random

import detection_data
import round_config
from detection_data import DetectionData, Annotation
import transforms
import glob

import trojai.datagen.utils
import trojai.datagen.blend_merges
import trojai.datagen.insert_merges
import trojai.datagen.image_entity
import trojai.datagen.xform_merge_pipeline
import trojai.datagen.static_color_xforms
import trojai.datagen.image_size_xforms
import trojai.datagen.image_affine_xforms
import trojai.datagen.noise_xforms
import trojai.datagen.albumentations_xforms
import trojai.datagen.transform_interface


from pycocotools import mask as mask_utils





logger = logging.getLogger()

trojan_instances_counters: Dict[int, multiprocessing.Value] = None
current_image_id: multiprocessing.Value = None
per_process_config: round_config.RoundConfig = None
foreground_merge_pipeline_obj: trojai.datagen.xform_merge_pipeline.XFormMerge = None
fg_xforms: Iterable[trojai.datagen.transform_interface.ImageTransform] = None
pipeline_obj: trojai.datagen.xform_merge_pipeline.XFormMerge = None
combined_xforms: Iterable[trojai.datagen.transform_interface.ImageTransform] = None


IMAGE_BUILD_NUMBER_TRIES = 20


def init_worker(config, trojan_instances_counters_arg):
    global per_process_config
    global trojan_instances_counters

    per_process_config = config
    trojan_instances_counters = trojan_instances_counters_arg

    # merge foreground into background
    global foreground_merge_pipeline_obj
    # handle the different insertion mechanisms based on task
    if per_process_config.task_type == round_config.CLASS:
        fg_merge_object = trojai.datagen.insert_merges.InsertRandomWithMask()
    elif per_process_config.task_type == round_config.OBJ:
        fg_merge_object = trojai.datagen.insert_merges.InsertRandomWithMaskPciam()
    else:
        raise RuntimeError("Invalid task type: {}".format(per_process_config.task_type))
    foreground_merge_pipeline_obj = trojai.datagen.xform_merge_pipeline.XFormMerge([[[], []]],
                                                                                   [fg_merge_object],
                                                                                   [])

    # specify the foreground xforms
    global fg_xforms
    fg_size_min = per_process_config.foreground_size_pixels_min
    fg_size_max = per_process_config.foreground_size_pixels_max
    fg_xforms = SyntheticImageDataset.build_foreground_xforms(per_process_config, fg_size_min, fg_size_max)

    # merge the finalized foregrounds into the background in a single step.
    global pipeline_obj
    # specify the foreground/background merge object
    merge_obj = trojai.datagen.blend_merges.BrightnessAdjustGrainMergePaste(lighting_adjuster=trojai.datagen.lighting_utils.adjust_brightness_mmprms)
    # specify the background xforms
    bg_xforms = SyntheticImageDataset.build_background_xforms(per_process_config)
    pipeline_obj = trojai.datagen.xform_merge_pipeline.XFormMerge([[bg_xforms, []]],
                                                                  [merge_obj],
                                                                  [])

    # specify the xforms for the final image
    global combined_xforms
    combined_xforms = SyntheticImageDataset.build_combined_xforms(per_process_config)


def build_image_worker(lcl_config: round_config.RoundConfig, rso: np.random.RandomState, fg_image_fps: list, bg_image_fp: str, bg_mask_fp: str, lcl_fg_merge_pipeline_obj, lcl_fg_xforms, lcl_pipeline_obj):
    try:
        # load background image
        bg_mask = cv2.imread(bg_mask_fp, cv2.IMREAD_UNCHANGED)

        processed_img = trojai.datagen.image_entity.GenericImageEntity(cv2.cvtColor(cv2.imread(bg_image_fp, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGB), semantic_label_mask=bg_mask)

        # Merge all foreground objects into background
        final_img_shape = (lcl_config.img_size_pixels, lcl_config.img_size_pixels, 4)
        processed_fg = trojai.datagen.image_entity.GenericImageEntity(np.zeros(final_img_shape, dtype=np.uint8))
        available_fg_mask = np.ones((lcl_config.img_size_pixels, lcl_config.img_size_pixels), dtype=np.uint8)
        fg_labeled_mask = np.zeros((lcl_config.img_size_pixels, lcl_config.img_size_pixels), dtype=np.uint8)

        for i in range(len(fg_image_fps)):
            fg_image_fp = fg_image_fps[i]

            # load foreground image
            sign_img = cv2.cvtColor(cv2.imread(fg_image_fp, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
            sign_mask = (sign_img[:, :, 3] > 0).astype(bool)
            fg_entity = trojai.datagen.image_entity.GenericImageEntity(sign_img, sign_mask)

            # apply any foreground xforms
            fg_entity = trojai.datagen.utils.process_xform_list(fg_entity, lcl_fg_xforms, rso)

            processed_fg = lcl_fg_merge_pipeline_obj.process([processed_fg, fg_entity], rso)
            # get the mask to determine where we have already inserted objects
            fg_mask = processed_fg.get_mask().astype(np.uint8)

            available_fg_mask[fg_mask > 0] = 0
            fg_labeled_mask[fg_mask > 0] = i + 1  # +1 offset to allow the background to have id=0

            processed_fg.mask = available_fg_mask

        # merge the finalized foregrounds into the background in a single step.
        processed_img = lcl_pipeline_obj.process([processed_img, processed_fg], rso)
        # processed_img.show_semantics()

        all_present_flag = verify_all_squential_labels_present(fg_labeled_mask, nb_ids=len(fg_image_fps))
        if not all_present_flag:
            # During the image build, some foreground was completely covered by a later foreground.
            # restart building this image from scratch
            return None, None

        # remove small objects (less than 50 pixels) from the used foreground mask
        fg_labeled_mask, completely_deleted_blob = SyntheticImageDataset.filter_small_objects(fg_labeled_mask, size=50)
        if completely_deleted_blob:
            # filter small objects completely deleted one of the foreground object, so this build_image run is not valid. Retry
            return None, None

        # processed_img.show_image()
        # processed_img.show_semantics()
        return processed_img, fg_labeled_mask
    except:
        logger.debug(traceback.format_exc())
        logger.debug("build_image_worker threw an error in trojai.datagen, retrying image build.")
        return None, None


def insert_trigger_worker(det_data: DetectionData, lcl_config: round_config.RoundConfig, rso: np.random.RandomState, class_counts: list, trjn_instances_counters: Dict[int, multiprocessing.Value]):
    attempted_trigger_executor = None

    # attempt to apply triggers in a random order
    trigger_idx_list = list(range(len(lcl_config.triggers)))
    rso.shuffle(trigger_idx_list)
    for trigger_idx in trigger_idx_list:
        trigger_executor = lcl_config.triggers[trigger_idx]
        trigger_id = trigger_executor.trigger_id

        if not det_data.spurious and lcl_config._master_rso.uniform() < trigger_executor.spurious_trigger_fraction:
            if hasattr(trigger_executor, 'update_trigger_color_texture'):
                # rotate the trigger color and texture if possible (only certain trigger_executors have this function)
                trigger_executor.update_trigger_color_texture(rso)

            # TODO Create a full list of Annotation objects containing all of the spurious triggers in the detection_data
            success = trigger_executor.apply_spurious_trigger(det_data, rso)
            if success:
                attempted_trigger_executor = trigger_executor

        # Identify source Ids that need to be triggered
        counter = trjn_instances_counters[trigger_id]
        with counter.get_lock():
            value = counter.value

        perc_trojaned = float(value) / float(class_counts[trigger_executor.source_class])
        # if we have already created enough triggered data for this trigger
        # if trigger_executor.trigger_fraction is None, then there is no cap on the number of triggers
        if trigger_executor.trigger_fraction is not None and perc_trojaned >= trigger_executor.trigger_fraction:
            continue

        if not trigger_executor.is_valid(det_data):
            continue

        if hasattr(trigger_executor, 'update_trigger_color_texture'):
            # rotate the trigger color and texture if possible (only certain trigger_executors have this function)
            trigger_executor.update_trigger_color_texture(rso)

        attempted_trigger_executor = trigger_executor
        triggered_class_id = None
        for trigger_insertion_attempt in range(IMAGE_BUILD_NUMBER_TRIES):
            triggered_class_id = trigger_executor.apply_trigger(det_data, rso)
            # if the trigger is successfully inserted, break
            if triggered_class_id is not None:
                break
        # det_data.view_data()
        if triggered_class_id is not None:
            # TODO Create a full list of Annotation objects containing all of the triggers in the detection_data
            with trjn_instances_counters[trigger_id].get_lock():
                trjn_instances_counters[trigger_id].value += 1

        if det_data.poisoned:
            # Only apply 1 trigger executor per image, so return as soon as one insertion is successful
            return det_data, attempted_trigger_executor

    return det_data, attempted_trigger_executor


def verify_all_squential_labels_present(used_fg_mask, nb_ids):
    # ensure that no foregrounds completely covered a prior foreground
    mask_label_ids = np.unique(used_fg_mask).tolist()
    for i in range(nb_ids):
        if (i + 1) not in mask_label_ids:
            logger.debug("During addition of {} foregrounds, id:{} was completely covered by a subsequent foreground.".format(nb_ids, (i + 1)))
            return False
    return True


def build_image(rso: np.random.RandomState, image_id: int, fg_image_fps: list, bg_image_fp: str, bg_mask_fp: str, obj_class_labels: list, class_counts: list) -> (DetectionData):
    # TODO update all the documentation for this whole repo
    """
    Worker function to build all possible configurations from the round config. This function takes in the config and random state object and produces an image instance that is valid given the config.
    :param rso: the random state object to draw random numbers from
    :param fg_image_fp: the filepath to the foreground image
    :param bg_image_fp: the filepath to the background image
    :param bg_mask_fp: the filepath to the background mask indicating semantic labels
    :param obj_class_label: the class label associated with the foreground filepath
    :return: Tuple (np.ndarray, int, int, bool)
    Image instance created from the specified foreground and background that is consistent with the round config.
    Training object class label (may be different from the foreground class if a trigger has been inserted.
    Foreground object class label.
    Flag indicating whether a class changing trigger has been inserted, i.e. whether the image has been poisoned.
    """
    global trojan_instances_counters
    # global spurious_instances
    global per_process_config
    global foreground_merge_pipeline_obj, fg_xforms, pipeline_obj, combined_xforms

    start_time = time.time()
    det_data = None
    # attempt to build the image N times, to account for potential construction failures
    for img_build_try_idx in range(IMAGE_BUILD_NUMBER_TRIES):
        det_data = None
        processed_img, fg_labeled_mask = build_image_worker(per_process_config, rso, fg_image_fps, bg_image_fp, bg_mask_fp, foreground_merge_pipeline_obj, fg_xforms, pipeline_obj)

        if processed_img is None:
            # build_image_worker threw an error (likely in the trojai datagen), continue and try again
            continue

        # view_mat(processed_img.get_data(), 'img')
        det_data = DetectionData(image_id, processed_img.get_data().astype(per_process_config.img_type), processed_img.get_semantic_label_mask())

        # package masks into annotations and semantic masks for DetectionData
        for i in range(len(obj_class_labels)):
            mask = fg_labeled_mask == i+1  # +1 offset to allow the background to have id=0
            # view_mat(mask, 'mask{}'.format(i))
            class_id = obj_class_labels[i]

            encoded_mask = Annotation.encode_mask(mask)
            bbox = mask_utils.toBbox(encoded_mask)
            area = mask_utils.area(encoded_mask)

            det_data.add_annotation(int(class_id), bbox.tolist(), encoded_mask, int(area))

        # TODO in detection_data, per trigger save a box and mask for where each trigger is in the final image

        # det_data.view_data(draw_bboxes=True)
        trigger_executor = None
        if per_process_config.poisoned:
            # modifies det_data
            det_data, trigger_executor = insert_trigger_worker(det_data, per_process_config, rso, class_counts, trojan_instances_counters)

        # det_data.view_data(title_prefix='before combined ', show_annotations=True, draw_bboxes=True)

        # Apply combined xforms
        final_entity = trojai.datagen.image_entity.GenericImageEntity(det_data.get_image_data())
        final_entity = trojai.datagen.utils.process_xform_list(final_entity, combined_xforms, rso)
        det_data.update_image_data(final_entity.get_data())

        if trigger_executor is not None and (det_data.poisoned or det_data.spurious):
            # apply trigger post-processing transforms
            trigger_executor.post_process_trigger(det_data, rso)

        if trigger_executor is not None and (not det_data.poisoned and not det_data.spurious):
            # trigger failed to take, if we have more tries left in the image build, retry, otherwise
            if (img_build_try_idx+1) >= IMAGE_BUILD_NUMBER_TRIES:  # handle 0 vs len difference
                logger.info("  injecting trigger into image_id={} failed.".format(image_id))
            continue
        else:
            # trigger took, so we break out of the image creation/trojaning loop
            logger.debug("  building image_id={} took: {}".format(image_id, time.time() - start_time))
            # image successfully built
            return det_data

    if det_data is None:
        # image build was unsuccessful
        raise RuntimeError("Building image_id={} failed. Since this would result in the dataset not having the required number of instances; dataset creation is terminating.".format(image_id))
    return det_data


class SyntheticImageDataset(torch.utils.data.Dataset):
    def __init__(self, name: str, config: round_config.RoundConfig,
                 rso: np.random.RandomState,
                 num_samples_to_generate: int,
                 dataset_dirpath: str,
                 augmentation_transforms):
        self.name = name
        self.config = config
        self._rso = rso
        self.num_samples_to_generate = num_samples_to_generate
        self.dataset_dirpath = dataset_dirpath
        self.augmentation_transforms = augmentation_transforms

        # self.available_foregrounds_filepath = foreground_images_filepath
        self.foregrounds_dirpath = os.path.join(config.output_filepath, 'foregrounds')
        self.foreground_image_format = 'png'
        # self.available_backgrounds_filepath = background_images_filepath
        self.background_image_format = 'jpg'
        self.backgrounds_dirpath = os.path.join(self.dataset_dirpath, 'backgrounds', config.source_dataset)
        bg_filenames = [fn for fn in os.listdir(self.backgrounds_dirpath) if fn.endswith(self.background_image_format)]
        self.number_background_images = int(len(bg_filenames))
        self.backgrounds_mask_dirpath = os.path.join(self.dataset_dirpath, 'backgrounds', config.source_dataset + "_masks")

        self.max_class_instance_count_per_image = int(config.max_class_instance_count_per_image.split(':')[1])
        self.max_class_count_per_image = int(config.max_class_count_per_image.split(':')[1])

        self.all_detection_data = list()
        self.all_poisoned_data = list()
        self.all_clean_data = list()

        n_if_every_instance_was_alone_in_the_image = (self.max_class_count_per_image * self.num_samples_to_generate) / self.config.number_classes
        if round_config.RoundConfig.MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS >= n_if_every_instance_was_alone_in_the_image:
            logger.warning("This dataset configuration is asking for a max of {} class instances per image and a total of {} images. Given the MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS threshold={}, consider that with this config; if the class instances were alone in the images (and class balanced), there would only be {} instances of each class. This round_config dataset request cannot be satisfied due to a conflict between MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS={} and the number of requested datapoints={} and the number of classes={}.".format(self.max_class_count_per_image, self.num_samples_to_generate, round_config.RoundConfig.MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS, n_if_every_instance_was_alone_in_the_image, round_config.RoundConfig.MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS, self.num_samples_to_generate, self.config.number_classes))
            raise RuntimeError('!!!! Please check your round_config setup. This round_config dataset request cannot be satisfied due to a conflict between MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS={} and the number of requested datapoints={} and the number of classes={}.'.format(round_config.RoundConfig.MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS, self.num_samples_to_generate, self.config.number_classes))


    def save_json(self, filepath: str):
        if not filepath.endswith('.json'):
            raise RuntimeError("Expecting a file ending in '.json'")
        try:
            with open(filepath, mode='w', encoding='utf-8') as f:
                f.write(jsonpickle.encode(self, warn=True, indent=2))
        except:
            msg = 'Failed writing file "{}".'.format(filepath)
            logger.warning(msg)
            raise

    @staticmethod
    def load_json(filepath: str):
        if not os.path.exists(filepath):
            raise RuntimeError("Filepath does not exists: {}".format(filepath))
        if not filepath.endswith('.json'):
            raise RuntimeError("Expecting a file ending in '.json'")
        try:
            with open(filepath, mode='r', encoding='utf-8') as f:
                obj = jsonpickle.decode(f.read())

        except json.decoder.JSONDecodeError:
            logging.error("JSON decode error for file: {}, is it a proper json?".format(filepath))
            raise
        except:
            msg = 'Failed reading file "{}".'.format(filepath)
            logger.warning(msg)
            raise

        return obj

    def serialize(self, filepath: str):
        """
        Function serializes this dataset object to disk in a human readable format.
        :param filepath: The filepath to a directory where the outputs will be written.
        """

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        for det_data in self.all_detection_data:
            det_data = copy.deepcopy(det_data)

            image_id = det_data.image_id
            fldr = 'id-{:08d}'.format(image_id)
            ofp = os.path.join(filepath, fldr)
            if not os.path.exists(ofp):
                os.makedirs(ofp)

            det_data.write_image(os.path.join(ofp, 'img.jpg'))
            det_data.write_combined_labeled_mask(os.path.join(ofp, 'mask.tif'))
            # TODO add serialization of trigger/spurious mask and boxes. I.e. create a full list of Annotation objects containing all of the triggers in the detection_data

            # blank out mask info, since thats stored in the global mask.tif
            for a in det_data._annotations:
                a.encoded_mask = None
            # blank out the compress image data, since thats stored in the img.jpg
            det_data._compressed_image_data = None

            det_data.save_json(os.path.join(ofp, 'detection_data.json'))

        obj_to_jsonize = copy.deepcopy(self)
        obj_to_jsonize.all_detection_data = None
        obj_to_jsonize.all_poisoned_data = None
        obj_to_jsonize.all_clean_data = None
        obj_to_jsonize.config = None
        obj_to_jsonize.augmentation_transforms = None
        ofp = os.path.join(filepath, 'dataset.json')
        try:
            with open(ofp, mode='w', encoding='utf-8') as f:
                f.write(jsonpickle.encode(obj_to_jsonize, warn=True, indent=2))
        except:
            msg = 'Failed writing file "{}".'.format(filepath)
            logger.warning(msg)
            raise

    @staticmethod
    def deserialize(filepath: str, config: round_config.RoundConfig, augmentation_transforms):
        ds = SyntheticImageDataset.load_json(os.path.join(filepath, 'dataset.json'))
        ds.config = config
        ds.all_detection_data = list()
        ds.all_poisoned_data = list()
        ds.all_clean_data = list()
        # augmentation_transforms don't serialize properly, so they need to be passed back in
        ds.augmentation_transforms = augmentation_transforms

        fldrs = [f for f in os.listdir(filepath) if f.startswith('id-')]
        fldrs.sort()
        for f in fldrs:
            ofp = os.path.join(filepath, f)

            img = detection_data.DetectionData.read_image(os.path.join(ofp, 'img.jpg'))
            mask = detection_data.DetectionData.read_combined_labeled_mask(os.path.join(ofp, 'mask.tif'))
            det_data = detection_data.DetectionData.load_json(filepath=os.path.join(ofp, 'detection_data.json'))

            det_data.update_image_data(img)

            for a_idx in range(len(det_data._annotations)):
                a = det_data._annotations[a_idx]
                a.encoded_mask = a.encode_mask(mask == a_idx+1)
            ds.all_detection_data.append(det_data)
            if det_data.poisoned:
                ds.all_poisoned_data.append(det_data)
            else:
                ds.all_clean_data.append(det_data)

        return ds

    def clean_poisoned_split(self):
        clean_dataset = type(self)(self.name + '_clean', self.config, self._rso, self.num_samples_to_generate, self.dataset_dirpath, self.augmentation_transforms)
        poisoned_dataset = type(self)(self.name + '_poisoned', self.config, self._rso, self.num_samples_to_generate, self.dataset_dirpath, self.augmentation_transforms)

        clean_dataset.all_detection_data.extend(self.all_clean_data)
        poisoned_dataset.all_detection_data.extend(self.all_poisoned_data)

        return clean_dataset, poisoned_dataset

    def get_poisoned_split(self):
        poisoned_dataset = type(self)(self.name + '_poisoned', self.config, self._rso, self.num_samples_to_generate, self.dataset_dirpath, self.augmentation_transforms)

        poisoned_dataset.all_detection_data.extend(self.all_poisoned_data)

        return poisoned_dataset

    def __len__(self):
        return len(self.all_detection_data)

    @staticmethod
    def build_background_xforms(config: round_config.RoundConfig):
        """
        Defines the chain of transformations which need to be applied to each background image.
        :return: list of trojai.datagen transformations
        """
        img_size = (config.img_size_pixels, config.img_size_pixels)

        bg_xforms = list()
        bg_xforms.append(trojai.datagen.static_color_xforms.RGBtoRGBA())
        bg_xforms.append(trojai.datagen.image_size_xforms.RandomSubCrop(new_size=img_size))
        return bg_xforms

    @staticmethod
    def filter_small_objects(labeled_mask: np.ndarray, size: int = 50) -> (np.ndarray, bool):
        if not issubclass(labeled_mask.dtype.type, np.integer):
            raise RuntimeError("Input must be integer typed to filter small objects.")

        max_id = np.max(labeled_mask)
        completely_deleted_blob = False
        for i in range(max_id):
            mask = labeled_mask == i + 1  # +1 offset to allow the background to have id=0, and cover max_id
            mask = mask.astype(np.uint8)
            _, label_ids, stats, _ = cv2.connectedComponentsWithStats(mask, 4)
            deleted_blobs = np.zeros((stats.shape[0]), dtype=np.int)
            for k in range(stats.shape[0]):
                area = stats[k, cv2.CC_STAT_AREA]
                if area < size:
                    deleted_blobs[k] = 1
                    labeled_mask[label_ids == k] = 0
            # if all (or all but one (background)) blob is deleted, warn the caller
            if np.count_nonzero(deleted_blobs) >= (deleted_blobs.size - 1):
                logger.debug("Object {} from labeled_mask would be completely deleted, as all of its area was in blobs <{} pixels".format(i + 1, size))
                completely_deleted_blob = True
        return labeled_mask, completely_deleted_blob

    @staticmethod
    def build_foreground_xforms(config: round_config.RoundConfig, fg_size_min, fg_size_max):
        """
        Defines the chain of transformations which need to be applied to each foreground image.
        :return: list of trojai.datagen transformations
        """
        img_size = (config.img_size_pixels, config.img_size_pixels)

        min_foreground_scale = float(fg_size_min) / float(img_size[0])
        max_foreground_scale = float(fg_size_max) / float(img_size[0])

        scale = (min_foreground_scale, max_foreground_scale)
        ratio = (0.6, 1.4)


        sign_xforms = list()
        sign_xforms.append(trojai.datagen.static_color_xforms.RGBtoRGBA())
        # resize foreground to output image size, so the percentage sizes in RandomResizeBeta are in relative to the final output, not whatever size the foreground happened to be
        sign_xforms.append(trojai.datagen.image_size_xforms.Resize(new_size=img_size))
        p = 30  # pad the foreground to allow rotation without clipping out the edge of the object
        sign_xforms.append(trojai.datagen.image_size_xforms.Pad((p,p,p,p)))
        sign_xforms.append(trojai.datagen.image_affine_xforms.RandomPerspectiveXForm(None))
        sign_xforms.append(trojai.datagen.image_size_xforms.RandomResizeBeta(scale=scale, ratio=ratio))

        return sign_xforms

    @staticmethod
    def build_combined_xforms(config: round_config.RoundConfig):
        """
        Defines the chain of transformations which need to be applied to each image after the foreground has been inserted into the background.
        :return: list of trojai.datagen transformations
        """
        # create the merge transformations for blending the foregrounds and backgrounds together
        combined_xforms = list()
        combined_xforms.append(trojai.datagen.noise_xforms.RandomGaussianBlurXForm(ksize_min=config.gaussian_blur_ksize_min, ksize_max=config.gaussian_blur_ksize_max))

        combined_xforms.append(trojai.datagen.albumentations_xforms.AddFogXForm(probability=config.fog_probability))
        combined_xforms.append(trojai.datagen.albumentations_xforms.AddRainXForm(probability=config.rain_probability))

        combined_xforms.append(trojai.datagen.static_color_xforms.RGBAtoRGB())
        return combined_xforms

    def poison_existing(self):
        self.all_poisoned_data = list()
        self.all_clean_data = list()

        merge_combined_xforms = self.build_combined_xforms(self.config)

        class_counts = list()
        for k in range(self.config.number_classes):
            class_counts.append(0)

        for det_data in self.all_detection_data:
            for a in det_data.get_annotations():
                class_counts[a.class_id] += 1

        trojan_counters = {}
        # Setup counters for each trigger
        for trigger in self.config.triggers:
            trigger_id = trigger.trigger_id
            if trigger_id not in trojan_counters:
                trojan_counters[trigger_id] = []
            trojan_counters[trigger_id] = multiprocessing.Value('i', 0)

        for det_idx in range(len(self.all_detection_data)):
            det_data = copy.deepcopy(self.all_detection_data[det_idx])

            for img_build_try_idx in range(IMAGE_BUILD_NUMBER_TRIES):
                # det_data.view_data(draw_bboxes=True)
                trigger_executor = None
                if self.config.poisoned:
                    # modifies det_data
                    det_data, trigger_executor = insert_trigger_worker(det_data, self.config, self._rso, class_counts, trojan_counters)

                # det_data.view_data(title_prefix='before combined ', show_annotations=True, draw_bboxes=True)

                # Apply combined xforms
                final_entity = trojai.datagen.image_entity.GenericImageEntity(det_data.get_image_data())
                final_entity = trojai.datagen.utils.process_xform_list(final_entity, merge_combined_xforms, self._rso)
                det_data.update_image_data(final_entity.get_data())

                if trigger_executor is not None and (det_data.poisoned or det_data.spurious):
                    # apply trigger post-processing transforms
                    trigger_executor.post_process_trigger(det_data)

                    # image successfully built
                    self.all_detection_data[det_idx] = det_data
                    self.all_poisoned_data.append(det_data)
                else:
                    # trigger not inserted, for whatever reason (could be we already have enough poisoned data)
                    self.all_clean_data.append(self.all_detection_data[det_idx])

    def build_class_instance_distribution(self, class_list=None):
        # attempt to build a valid distribution of class instances for this dataset
        for class_instance_distribution_idx in range(IMAGE_BUILD_NUMBER_TRIES):
            if class_list is None:
                # unless requested otherwise, build images for all classes
                class_list = list(range(0, self.config.number_classes))
            obj_class_list = list()

            for i in range(self.num_samples_to_generate):
                # Randomly select the number of classes and the number of instances in each image based on Beta distribution
                image_class_list = []
                number_classes = int(np.ceil(self._rso.beta(2, 4) * self.max_class_count_per_image))
                number_classes = min(number_classes, len(class_list))
                class_ids = self._rso.choice(class_list, number_classes, replace=False)
                for c in class_ids:
                    number_instances = int(np.ceil(self._rso.beta(1, 4) * self.max_class_instance_count_per_image))
                    image_class_list.extend([int(c)] * number_instances)
                self._rso.shuffle(image_class_list)
                if len(image_class_list) > self.config.max_total_class_count_per_image:
                    image_class_list = image_class_list[:self.config.max_total_class_count_per_image]

                obj_class_list.append(image_class_list)

            class_counts = {}
            for class_id in class_list:
                class_counts[class_id] = 0

            for temp_class_list in obj_class_list:
                for class_id in temp_class_list:
                    class_counts[class_id] += 1

            # confirm that the distribution of class ids is valid
            valid_class_distribution = True
            for class_id in class_counts.keys():
                if class_counts[class_id] < round_config.RoundConfig.MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS:
                    logger.debug("  creation of class distribution per image is invalid. Try ({}/{}). Not enough instances ({} < {}) for Class ID {}".format(class_instance_distribution_idx, IMAGE_BUILD_NUMBER_TRIES, class_counts[class_id], round_config.RoundConfig.MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS, class_id))
                    valid_class_distribution = False

            if valid_class_distribution:
                return obj_class_list, class_counts

        raise RuntimeError('Failed to build a valid class instance distribution after {} tries. Please check your config parameters.'.format(IMAGE_BUILD_NUMBER_TRIES))

    def build_dataset(self, class_list=None, verify_trojan_fraction=True):
        global trojan_instances_counters
        logger.info('Building synthetic image dataset ({}): {} samples'.format(self.name, self.num_samples_to_generate))
        start_time = time.time()

        # get listing of all bg files
        bg_image_fps = glob.glob(os.path.join(self.backgrounds_dirpath, '**', '*.' + self.background_image_format), recursive=True)
        # enforce deterministic background order
        bg_image_fps.sort()
        self._rso.shuffle(bg_image_fps)

        bg_mask_fps = []
        for fp in bg_image_fps:
            parent, fn = os.path.split(fp)
            parent = parent.replace(self.backgrounds_dirpath, self.backgrounds_mask_dirpath)
            toks = fn.split('_')
            fn = "_".join(toks[0:3]) + "_gtCoarse_labelIds.png"
            bg_mask_fps.append(os.path.join(parent, fn))

        # get listing of all foreground images
        fg_image_fps = glob.glob(os.path.join(self.foregrounds_dirpath, '**', '*.' + self.foreground_image_format), recursive=True)
        # enforce deterministic foreground order, which equates to class label mapping
        fg_image_fps.sort()
        fg_class_id_translation_dict = dict()
        for i in range(len(fg_image_fps)):
            _, fn = os.path.split(fg_image_fps[i])
            fg_class_id_translation_dict[i] = fn
        with open(os.path.join(self.config.output_filepath, 'fg_class_translation.json'), 'w') as fh:
            json.dump(fg_class_id_translation_dict, fh, ensure_ascii=True, indent=2)
        if self.config.number_classes != len(fg_image_fps):
            raise RuntimeError("config.number_classes={} and len(fg_image_fps)={} disagree, which is impossible.".format(self.config.number_classes, len(fg_image_fps)))

        # build the distribution of class instances per image for this dataset
        obj_class_list, class_counts = self.build_class_instance_distribution(class_list)

        for class_id, count in class_counts.items():
            self.config.total_class_instances[class_id] = count

        logger.info('Using {} CPU core(s) to generate data'.format(self.config.num_workers))

        worker_input_list = list()
        for i in range(len(obj_class_list)):
            temp_class_list = obj_class_list[i]
            sign_image_fps = list()
            for class_id in temp_class_list:
                sign_image_fps.append(fg_image_fps[class_id])

            bg_image_idx = self._rso.randint(low=0, high=len(bg_image_fps))
            bg_image_f = bg_image_fps[bg_image_idx]
            bg_mask_f = bg_mask_fps[bg_image_idx]

            rso = np.random.RandomState(self._rso.randint(2 ** 31 - 1))
            worker_input_list.append((rso, i, sign_image_fps, bg_image_f, bg_mask_f, temp_class_list, class_counts))

        trojan_counters = {}
        # Setup counters for each trigger

        for trigger in self.config.triggers:
            trigger_id = trigger.trigger_id
            if trigger_id not in trojan_counters:
                trojan_counters[trigger_id] = []
            trojan_counters[trigger_id] = multiprocessing.Value('i', 0)

        trojan_instances_counters = trojan_counters

        if self.config.num_workers == 0:
            # match the APIs for multiprocess pool, but do all the work on the master thread
            # init_worker(self.config, trojan_counters, current_image_id_counter)  # setup the global variables
            init_worker(self.config, trojan_counters)  # setup the global variables
            results = list()
            for args in worker_input_list:
                results.append(build_image(*args))
        else:
            with multiprocessing.Pool(processes=self.config.num_workers, initializer=init_worker,
                                       initargs=(self.config, trojan_counters)) as pool:
                # perform the work in parallel
                results = pool.starmap(build_image, worker_input_list)

        # Verify trojaning was inserted based on the trigger fraction
        for trigger_executor in self.config.triggers:
            trigger_id = trigger_executor.trigger_id
            source_class = trigger_executor.source_class
            trigger_fraction = trigger_executor.trigger_fraction

            num_instances = trojan_instances_counters[trigger_id].value
            total_instances = class_counts[source_class]
            cur_fraction = float(num_instances) / float(total_instances)

            trigger_executor.update_actual_trigger_fraction(self.name, source_class, num_instances, cur_fraction)

            if verify_trojan_fraction:
                if trigger_fraction is None:
                    if cur_fraction == 0.0:
                        msg = 'Invalid trigger percentage after trojaning for trigger_id: {},\n' \
                              'Source trigger class: {}. Target trojan percentage = {} (all valid images), actual trojan percentage = {}'.format(
                            trigger_id, source_class, trigger_fraction, cur_fraction)
                        logging.error(msg)
                        raise RuntimeError(msg)
                else:
                    if cur_fraction < trigger_fraction:
                        msg = 'Invalid trigger percentage after trojaning for trigger_id: {},\n' \
                              'Source trigger class: {}. Target trojan percentage = {}, actual trojan percentage = {}'.format(
                            trigger_id, source_class, trigger_fraction, cur_fraction)
                        logging.error(msg)
                        raise RuntimeError(msg)

        # shuffle resutls, so all the poisoned/spurious data is not at the front
        self._rso.shuffle(results)
        for result in results:
            if results is None:
                logging.error("Image Dataset construction failed.")
                raise RuntimeError("Image Dataset construction failed.")
            self.all_detection_data.append(result)

            if result.poisoned:
                self.all_poisoned_data.append(result)
            else:
                self.all_clean_data.append(result)

        elapsed_time = time.time() - start_time
        logger.info("{} dataset construction took {}s".format(self.name, elapsed_time))
        logger.info("  {} cpu/seconds per image".format(elapsed_time/float(len(self.all_detection_data))))

    def dump_jpg_examples(self, clean=True, n=20, spurious=False):
        fldr = '{}-clean-example-data'.format(self.name)
        dn = self.all_clean_data
        if not clean:
            fldr = '{}-poisoned-example-data'.format(self.name)
            dn = self.all_poisoned_data
            dn = [d for d in dn if not d.spurious]
        else:
            dn = [d for d in dn if not d.spurious]
        if spurious:
            dn = self.all_clean_data
            fldr = '{}-spurious-clean-example-data'.format(self.name)
            dn = [d for d in dn if d.spurious]

        ofp = os.path.join(self.config.output_filepath, fldr)
        if os.path.exists(ofp):
            start_idx = len([fn for fn in os.listdir(ofp) if fn.endswith(".jpg")])
        else:
            start_idx = 0
        end_idx = min(n, len(dn))
        if end_idx < n:
            logger.info("dump_jpg_examples only has {} instances available, instead of the requested {}.".format(end_idx, n))

        idx_list = list(range(start_idx, end_idx))
        if len(idx_list) > 0:
            if not os.path.exists(ofp):
                os.makedirs(ofp)

        for i in idx_list:
            det_data = dn[i]
            det_data.view_data(draw_bboxes=True, output_filepath=os.path.join(ofp, 'img-{:08}.jpg'.format(det_data.image_id)))

    def build_examples(self, output_filepath, num_samples_per_class):
        raise RuntimeError('Not Implemented')






class ClassificationDataset(SyntheticImageDataset):
    train_augmentation_transforms = torchvision.transforms.Compose(
        [
            transforms.RandomPhotometricDistort(),
            torchvision.transforms.RandomCrop(size=round_config.RoundConfig.BASE_IMAGE_SIZE, padding=int(10), padding_mode='reflect'),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ConvertImageDtype(torch.float),
        ]
    )

    test_augmentation_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ConvertImageDtype(torch.float),
        ]
    )

    def __init__(self, name: str, config: round_config.RoundConfig,
                 rso: np.random.RandomState,
                 num_samples_to_generate: int,
                 dataset_dirpath: str,
                 augmentation_transforms
                 ):
        super().__init__(name, config, rso, num_samples_to_generate, dataset_dirpath, augmentation_transforms)

    def __getitem__(self, item):
        det_data = self.all_detection_data[item]

        image_data = det_data.get_image_data(as_rgb=True)
        train_labels = det_data.get_class_label_list()

        if len(train_labels) != 1:
            raise RuntimeError('Classification expected 1 label, got {} labels'.format(len(train_labels)))

        train_label = torch.as_tensor(train_labels[0])
        image_data = torch.as_tensor(image_data)  # should be uint8 type, the conversion to float is handled later
        # move channels first
        image_data = image_data.permute((2, 0, 1))

        image_data = self.augmentation_transforms(image_data)

        return image_data, train_label


class ObjectDetectionDataset(SyntheticImageDataset):

    train_augmentation_transforms = transforms.ObjCompose(
        [
            transforms.RandomPhotometricDistortObjDet(),
            transforms.RandomIoUCrop(),
            transforms.Resize((round_config.RoundConfig.BASE_IMAGE_SIZE, round_config.RoundConfig.BASE_IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ConvertImageDtype(torch.float),
        ]
    )

    test_augmentation_transforms = transforms.ObjCompose(
        [
            transforms.ConvertImageDtype(torch.float),
        ]
    )

    def __init__(self, name: str, config: round_config.RoundConfig,
                 rso: np.random.RandomState,
                 num_samples_to_generate: int,
                 dataset_dirpath: str,
                 augmentation_transforms
                 ):
        super().__init__(name, config, rso, num_samples_to_generate, dataset_dirpath, augmentation_transforms)

    def __getitem__(self, item):
        det_data = self.all_detection_data[item]

        image_data = det_data.get_image_data(as_rgb=True)
        annotations = det_data.get_annotations()
        boxes = det_data.get_box_list()
        class_ids = det_data.get_class_label_list()

        if len(boxes) == 0:
            boxes = np.zeros((0, 4))
        else:
            # convert boxes from list into numpy array
            boxes = [np.asarray(b) for b in boxes]
            boxes = np.stack(boxes)
            # convert [x,y,w,h] to [x0, y0, x1, y1]
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        target = {}
        target['boxes'] = torch.as_tensor(boxes)
        target['labels'] = torch.as_tensor(class_ids).type(torch.int64)

        image_data = torch.as_tensor(image_data)  # should be uint8 type, the conversion to float is handled later
        # move channels first
        image_data = image_data.permute((2, 0, 1))

        # from matplotlib import pyplot as plt
        # img = copy.deepcopy(image_data.detach().cpu().numpy()).squeeze()
        # b = copy.deepcopy(target['boxes'].detach().cpu().numpy())
        # img = img.transpose((1, 2, 0))
        # import bbox_utils
        # img = bbox_utils.draw_boxes(img, b, value=[255, 0, 0])
        # plt.imshow(img)
        # plt.show()

        image_data, target = self.augmentation_transforms(image_data, target)

        degenerate_boxes = (target['boxes'][:, 2:] - target['boxes'][:, :2]) < round_config.RoundConfig.MIN_BOX_DIMENSION
        degenerate_boxes = torch.sum(degenerate_boxes, dim=1)
        if degenerate_boxes.any():
            target['boxes'] = target['boxes'][degenerate_boxes == 0, :]
            target['labels'] = target['labels'][degenerate_boxes == 0]

        # from matplotlib import pyplot as plt
        # img = copy.deepcopy(image_data.detach().cpu().numpy()).squeeze()
        # b = copy.deepcopy(target['boxes'].detach().cpu().numpy())
        # img = img.transpose((1, 2, 0))
        # import bbox_utils
        # img = bbox_utils.draw_boxes(img, b, value=[255, 0, 0])
        # plt.imshow(img)
        # plt.show()

        return image_data, target



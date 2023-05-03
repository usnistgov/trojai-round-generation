import os
import multiprocessing
from typing import Iterable

import numpy as np
import json
import torch
import torchvision
import logging
import traceback
import time
import copy
import cv2
import transforms
import glob

# local imports
import base_config
import dataset
import detection_data
import dataset_utils

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


trojan_instances_counters: dict[int, multiprocessing.Value] = None
current_image_id: multiprocessing.Value = None
per_process_config: base_config.Config = None
foreground_merge_pipeline_obj: trojai.datagen.xform_merge_pipeline.XFormMerge = None
fg_xforms: Iterable[trojai.datagen.transform_interface.ImageTransform] = None
pipeline_obj: trojai.datagen.xform_merge_pipeline.XFormMerge = None
combined_xforms: Iterable[trojai.datagen.transform_interface.ImageTransform] = None




def init_worker(config: base_config.ConfigSynth, trojan_instances_counters_arg):
    """Worker initialization function for Synth-data work.
    This function is expected to be called per-worker at the start of multiprocessing loop

    Args:
        config
        trojan_instances_counters_arg
    """

    global per_process_config
    global trojan_instances_counters

    per_process_config = config
    trojan_instances_counters = trojan_instances_counters_arg

    # merge foreground into background
    global foreground_merge_pipeline_obj
    # handle the different insertion mechanisms based on task
    if per_process_config.TASK_TYPE == base_config.CLASS:
        fg_merge_object = trojai.datagen.insert_merges.InsertRandomWithMask()
    elif per_process_config.TASK_TYPE == base_config.OBJ:
        fg_merge_object = trojai.datagen.insert_merges.InsertRandomWithMaskPciam()
    else:
        raise RuntimeError("Invalid task type: {}".format(per_process_config.TASK_TYPE))
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


def verify_all_sequential_labels_present(used_fg_mask, nb_ids):
    # ensure that no foregrounds completely covered a prior foreground
    mask_label_ids = np.unique(used_fg_mask).tolist()
    for i in range(nb_ids):
        if (i + 1) not in mask_label_ids:
            logging.debug("During addition of {} foregrounds, id:{} was completely covered by a subsequent foreground.".format(nb_ids, (i + 1)))
            return False
    return True


def build_image_worker(lcl_config: base_config.ConfigSynth, rso: np.random.RandomState, fg_image_fps: list[str], bg_image_fp: str, bg_mask_fp: str, lcl_fg_merge_pipeline_obj, lcl_fg_xforms, lcl_pipeline_obj, data_split_name):
    """Worker function to build synth data images.
    This function constructs a single image and annotation pair.

    Args:
        lcl_config: a local copy of the governing round config.
        rso: the random state object controlling randomness in this function.
        fg_image_fps: list of image file names for the composite synth image foregrounds.
        bg_image_fp: filepath to the background image.
        bg_mask_fp: filepath to the background mask image containing the semantic information for the background image.
        lcl_fg_merge_pipeline_obj: trojai package merge object for the foreground pipeline.
        lcl_fg_xforms: trojai package transforms to apply to the foreground object.
        lcl_pipeline_obj: local pipeline object for combining the foreground and background.
        data_split_name: name for the datasplit being built. Controls the number of instances to build, where the train dataset builds all N instances, but val and test build only lcl_config.validation_split percent.
    """

    try:
        # load background image only if required
        if bg_mask_fp is None or not os.path.exists(bg_mask_fp):
            bg_mask = None
        else:
            bg_mask = cv2.imread(bg_mask_fp, cv2.IMREAD_UNCHANGED)
    except RuntimeError as e:
        logging.info(traceback.format_exc())
        logging.info("build_image_worker threw an error while loading the source images, retrying image build.")
        return None, None

    try:
        orig_processed_img = trojai.datagen.image_entity.GenericImageEntity(cv2.cvtColor(cv2.imread(bg_image_fp, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGB), semantic_label_mask=bg_mask)

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
        processed_img = lcl_pipeline_obj.process([orig_processed_img, processed_fg], rso)
        # processed_img.show_semantics()

        all_present_flag = verify_all_sequential_labels_present(fg_labeled_mask, nb_ids=len(fg_image_fps))
        if not all_present_flag:
            # During the image build, some foreground was completely covered by a later foreground.
            # restart building this image from scratch
            return None, None

        # remove small objects (less than 50 pixels) from the used foreground mask
        fg_labeled_mask, completely_deleted_blob = dataset_utils.filter_small_objects(fg_labeled_mask, size=50)
        if completely_deleted_blob:
            # filter small objects completely deleted one of the foreground object, so this build_image run is not valid. Retry
            return None, None

        # processed_img.show_image()
        # processed_img.show_semantics()
        return processed_img, fg_labeled_mask
    except RuntimeError as e:
        logging.info(traceback.format_exc())
        logging.info("build_image_worker threw an error in trojai.datagen, retrying image build.")
        return None, None


def build_image(rso: np.random.RandomState, image_id: int, fg_image_fps: list, bg_image_fp: str, bg_mask_fp: str, obj_class_labels: list[int], class_counts: list[int], data_split_name: str) -> detection_data.DetectionData:
    """Worker function to build all possible configurations from the round config. This function takes in the config and random state object and produces an image instance that is valid given the config.

    Args:
        rso: the random state object to draw random numbers from
        fg_image_fp: the filepath to the foreground image
        bg_image_fp: the filepath to the background image
        bg_mask_fp: the filepath to the background mask indicating semantic labels
        obj_class_label: the class label associated with the foreground filepath
        class_counts: the number of instances of each class in the dataset.
        data_split_name: the name of the data split being constructed.

    :return: DetectionData instance holding the image and annotation data.
    """
    global trojan_instances_counters
    global per_process_config
    global foreground_merge_pipeline_obj, fg_xforms, pipeline_obj, combined_xforms

    start_time = time.time()
    det_data = None
    # attempt to build the image N times, to account for potential construction failures
    for img_build_try_idx in range(dataset.IMAGE_BUILD_NUMBER_TRIES):
        det_data = None
        processed_img, fg_labeled_mask = build_image_worker(per_process_config, rso, fg_image_fps, bg_image_fp, bg_mask_fp, foreground_merge_pipeline_obj, fg_xforms, pipeline_obj, data_split_name)

        if processed_img is None:
            # build_image_worker threw an error (likely in the trojai datagen), continue and try again
            continue

        # view_mat(processed_img.get_data(), 'img')
        det_data = detection_data.DetectionData(image_id, processed_img.get_data().astype(per_process_config.img_type), processed_img.get_semantic_label_mask())

        # package masks into annotations and semantic masks for DetectionData
        for i in range(len(obj_class_labels)):
            mask = fg_labeled_mask == i + 1  # +1 offset to allow the background to have id=0
            # view_mat(mask, 'mask{}'.format(i))
            class_id = obj_class_labels[i]

            det_data.add_annotation(int(class_id), ann_mask=mask)

        # TODO in detection_data, per trigger save a box and mask for where each trigger is in the final image

        # det_data.view_data(draw_bboxes=True)
        trigger_executor = None
        if per_process_config.poisoned:
            # modifies det_data
            det_data, trigger_executor = dataset_utils.insert_trigger_worker(det_data, per_process_config, rso, class_counts, trojan_instances_counters)

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
            if (img_build_try_idx + 1) >= dataset.IMAGE_BUILD_NUMBER_TRIES:  # handle 0 vs len difference
                logging.debug("  injecting trigger into image_id={} failed.".format(image_id))
            continue
        else:
            # trigger took, so we break out of the image creation/trojaning loop
            logging.debug("  building image_id={} took: {}".format(image_id, time.time() - start_time))
            # image successfully built
            return det_data

    if det_data is None:
        # image build was unsuccessful
        raise RuntimeError("Building image_id={} failed. Since this would result in the dataset not having the required number of instances; dataset creation is terminating.".format(image_id))
    return det_data


class SyntheticImageDataset(dataset.ImageDataset):
    """TrojAI Synthetic Image Dataset.

    Args:
        name (str): the name for this dataset, usually something like \"train_clean\" or \"val_poisoned\".
        config (base_config.Config): the configuration object governing this training instance.
        rso (np.random.RandomState): the random state object controlling any randomness used within this dataset
        dataset_dirpath (str): the absolute filepath to the directory containing the dataset. This is only used for non-synthetic datasets.
        augmentation_transforms: The augmentation transforms to apply within the __getitem__ function before returning the data instance.
    """

    def __init__(self, name: str, config: base_config.Config,
                 rso: np.random.RandomState,
                 num_samples_to_generate: int,
                 dataset_dirpath: str,
                 augmentation_transforms=None):
        super().__init__(name, config, rso, dataset_dirpath, augmentation_transforms)

        self.num_samples_to_generate = num_samples_to_generate

        self.foregrounds_dirpath = os.path.join(self.config.output_filepath, 'foregrounds')
        self.foreground_image_format = 'png'

        self.background_image_format = 'jpg'
        self.background_mask_format = 'png'
        self.backgrounds_dirpath = os.path.join(self.dataset_dirpath, self.config.source_dataset_background.value)
        bg_filenames = [fn for fn in os.listdir(self.backgrounds_dirpath) if fn.endswith(self.background_image_format)]
        self.number_background_images = int(len(bg_filenames))
        self.backgrounds_mask_dirpath = os.path.join(self.dataset_dirpath, self.config.source_dataset_background.value + "_masks")
        if not os.path.exists(self.backgrounds_mask_dirpath):
            self.backgrounds_mask_dirpath = None

        self.max_class_instance_count_per_image = int(self.config.max_class_instance_count_per_image.value)
        self.max_class_count_per_image = int(self.config.max_class_count_per_image.value)

        n_if_every_instance_was_alone_in_the_image = (self.max_class_count_per_image * self.num_samples_to_generate) / self.config.number_classes.value
        if self.config.MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS >= n_if_every_instance_was_alone_in_the_image:
            logging.warning("This dataset configuration is asking for a max of {} class instances per image and a total of {} images. Given the MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS threshold={}, consider that with this config; if the class instances were alone in the images (and class balanced), there would only be {} instances of each class. This config dataset request cannot be satisfied due to a conflict between MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS={} and the number of requested datapoints={} and the number of classes={}.".format(self.max_class_count_per_image, self.num_samples_to_generate, self.config.MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS, n_if_every_instance_was_alone_in_the_image, self.config.MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS, self.num_samples_to_generate, self.config.number_classes.value))
            raise RuntimeError('!!!! Please check your config setup. This config dataset request cannot be satisfied due to a conflict between MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS={} and the number of requested datapoints={} and the number of classes={}.'.format(self.config.MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS, self.num_samples_to_generate, self.config.number_classes.value))

    def clean_poisoned_split(self):
        """Split the dataset into two, one containing just the clean data, and one containing just the poisoned data.
        """
        # override the parent class function to pass along the num_samples_to_generate parameter
        clean_dataset = type(self)(self.name + '_clean', self.config, self._rso, self.num_samples_to_generate, self.dataset_dirpath, self.augmentation_transforms)
        poisoned_dataset = type(self)(self.name + '_poisoned', self.config, self._rso, self.num_samples_to_generate, self.dataset_dirpath, self.augmentation_transforms)

        clean_dataset.all_detection_data.extend(self.all_clean_data)
        clean_dataset.all_clean_data.extend(self.all_clean_data)
        poisoned_dataset.all_detection_data.extend(self.all_poisoned_data)
        poisoned_dataset.all_poisoned_data.extend(self.all_poisoned_data)

        return clean_dataset, poisoned_dataset

    def get_poisoned_split(self):
        """Get just the poisoned data as a dataset
        """
        # override the parent class function to pass along the num_samples_to_generate parameter
        poisoned_dataset = type(self)(self.name + '_poisoned', self.config, self._rso, self.num_samples_to_generate, self.dataset_dirpath, self.augmentation_transforms)

        poisoned_dataset.all_detection_data.extend(self.all_poisoned_data)
        poisoned_dataset.all_poisoned_data.extend(self.all_poisoned_data)

        return poisoned_dataset

    @staticmethod
    def build_background_xforms(config: base_config.Config):
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
    def build_foreground_xforms(config: base_config.Config, fg_size_min, fg_size_max):
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
        sign_xforms.append(trojai.datagen.image_size_xforms.Pad((p, p, p, p)))
        sign_xforms.append(trojai.datagen.image_affine_xforms.RandomPerspectiveXForm(None))
        sign_xforms.append(trojai.datagen.image_size_xforms.RandomResizeBeta(scale=scale, ratio=ratio))

        return sign_xforms

    @staticmethod
    def build_combined_xforms(config: base_config.Config):
        """
        Defines the chain of transformations which need to be applied to each image after the foreground has been inserted into the background.

        :return: list of trojai.datagen transformations
        """
        # create the merge transformations for blending the foregrounds and backgrounds together
        combined_xforms = list()
        combined_xforms.append(trojai.datagen.noise_xforms.RandomGaussianBlurXForm(ksize_min=config.gaussian_blur_ksize_min, ksize_max=config.gaussian_blur_ksize_max))

        combined_xforms.append(trojai.datagen.albumentations_xforms.AddFogXForm(probability=config.fog_probability.value))
        combined_xforms.append(trojai.datagen.albumentations_xforms.AddRainXForm(probability=config.rain_probability.value))

        combined_xforms.append(trojai.datagen.static_color_xforms.RGBAtoRGB())
        return combined_xforms

    def poison_existing(self):
        """
        Utility function to poison an existing dataset object, instead of relying on the build_dataset function.
        This is intended to be used with the serialize and deserialize functions.
        """
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

            for img_build_try_idx in range(dataset.IMAGE_BUILD_NUMBER_TRIES):
                # det_data.view_data(draw_bboxes=True)
                trigger_executor = None
                if self.config.poisoned:
                    # modifies det_data
                    det_data, trigger_executor = dataset_utils.insert_trigger_worker(det_data, self.config, self._rso, class_counts, trojan_counters)

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

    def build_class_instance_distribution(self, class_list: list[int] = None):
        """
        Build a distribution of class instances per image based on the control parameters from the config.

        Args:
            class_list: list of class ids to build data for.
        """
        # attempt to build a valid distribution of class instances for this dataset
        for class_instance_distribution_idx in range(dataset.IMAGE_BUILD_NUMBER_TRIES):
            if class_list is None:
                # unless requested otherwise, build images for all classes
                class_list = list(range(0, self.config.number_classes.value))
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
                if len(image_class_list) > self.config.max_total_class_count_per_image.value:
                    image_class_list = image_class_list[:self.config.max_total_class_count_per_image.value]

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
                if class_counts[class_id] < self.config.MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS:
                    logging.debug("  creation of class distribution per image is invalid. Try ({}/{}). Not enough instances ({} < {}) for Class ID {}".format(class_instance_distribution_idx, dataset.IMAGE_BUILD_NUMBER_TRIES, class_counts[class_id], self.config.MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS, class_id))
                    valid_class_distribution = False

            if valid_class_distribution:
                return obj_class_list, class_counts

        raise RuntimeError('Failed to build a valid class instance distribution after {} tries. Please check your config parameters.'.format(dataset.IMAGE_BUILD_NUMBER_TRIES))

    def build_dataset(self, class_list: list[int] = None, verify_trojan_fraction: bool = True):
        """build_dataset refers to synthetic datasets where trojaning happens during dataset creation.
        If the dataset is being loaded from disk, use load_dataset and trojan.
        Functionality wise: build_dataset = load_dataset + trojan methods.

        Args:
            class_list: the set of class labels to build the dataset for. If None, all classes are build. A subset of class labels can be useful when creating subsets of the dataset for additional testing and evaluation.
            verify_trojan_fraction: flag controlling whether to verify the trojan percentage is as requested after dataset construction is complete.
        """

        global trojan_instances_counters
        logging.info('Building synthetic image dataset ({}): {} samples'.format(self.name, self.num_samples_to_generate))
        start_time = time.time()

        # get listing of all bg files
        bg_image_fps = [os.path.join(self.backgrounds_dirpath, fn) for fn in os.listdir(self.backgrounds_dirpath) if fn.endswith(self.background_image_format)]
        bg_image_fps.sort()
        self._rso.shuffle(bg_image_fps)

        if self.backgrounds_mask_dirpath is None:
            bg_mask_fps = None
        else:
            bg_mask_fps = []
            for fp in bg_image_fps:
                _, fn = os.path.split(fp)
                fn = fn[0:-len(self.background_image_format)] + self.background_mask_format
                bg_mask_fps.append(os.path.join(self.backgrounds_mask_dirpath, fn))

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
        self.config.fg_class_translation = fg_class_id_translation_dict
        if self.config.number_classes.value != len(fg_image_fps):
            raise RuntimeError("config.number_classes={} and len(fg_image_fps)={} disagree, which is impossible.".format(self.config.number_classes.value, len(fg_image_fps)))

        # build the distribution of class instances per image for this dataset
        obj_class_list, class_counts = self.build_class_instance_distribution(class_list)

        for class_id, count in class_counts.items():
            self.config.total_class_instances[class_id] = count

        logging.info('Using {} CPU core(s) to generate data'.format(self.config.num_workers))

        worker_input_list = list()
        for i in range(len(obj_class_list)):
            temp_class_list = obj_class_list[i]
            sign_image_fps = list()
            for class_id in temp_class_list:
                sign_image_fps.append(fg_image_fps[class_id])

            bg_image_idx = self._rso.randint(low=0, high=len(bg_image_fps))
            bg_image_f = bg_image_fps[bg_image_idx]
            if bg_mask_fps is None:
                bg_mask_f = None
            else:
                bg_mask_f = bg_mask_fps[bg_image_idx]

            rso = np.random.RandomState(self._rso.randint(2 ** 31 - 1))
            data_split_name = self.name
            worker_input_list.append((rso, i, sign_image_fps, bg_image_f, bg_mask_f, temp_class_list, class_counts, data_split_name))

        trojan_counters = {}
        # Setup counters for each trigger

        for trigger_executor in self.config.triggers:
            trigger_id = trigger_executor.value.trigger_id
            if trigger_id not in trojan_counters:
                trojan_counters[trigger_id] = []
            trojan_counters[trigger_id] = multiprocessing.Value('i', 0)

        trojan_instances_counters = trojan_counters

        if self.config.num_workers_datagen == 0:
            # match the APIs for multiprocess pool, but do all the work on the master thread
            # init_worker(self.config, trojan_counters, current_image_id_counter)  # setup the global variables
            init_worker(self.config, trojan_counters)  # setup the global variables
            results = list()
            for args in worker_input_list:
                results.append(build_image(*args))
        else:
            with multiprocessing.Pool(processes=self.config.num_workers_datagen, initializer=init_worker,
                                      initargs=(self.config, trojan_counters)) as pool:
                # perform the work in parallel
                results = pool.starmap(build_image, worker_input_list)

        # Verify trojaning was inserted based on the trigger fraction
        for trigger_executor in self.config.triggers:
            trigger_executor = trigger_executor.value
            trigger_id = trigger_executor.trigger_id
            source_class = trigger_executor.source_class

            num_instances = trojan_instances_counters[trigger_id].value
            total_instances = class_counts[source_class]
            cur_fraction = float(num_instances) / float(total_instances)

            trigger_executor.update_actual_trigger_fraction(self.name, source_class, num_instances, cur_fraction)

            # don't validate the trojan percentage for background context triggers
            if verify_trojan_fraction:
                if trigger_executor.trigger_fraction is None or trigger_executor.trigger_fraction.value is None or 'backgroundcontext' in trigger_executor.__class__.__name__.lower():
                    if cur_fraction == 0.0:
                        msg = 'Invalid trigger percentage after trojaning for trigger_id: {},\n' \
                              'Source trigger class: {}. Target trojan percentage = (all valid images), actual trojan percentage = {}'.format(
                            trigger_id, source_class, cur_fraction)
                        logging.error(msg)
                        raise RuntimeError(msg)
                else:
                    resolution = max(0.02, 1.0 / float(total_instances))  # 2% or an error of 1 annotation, whichever is bigger
                    if np.abs(cur_fraction - trigger_executor.trigger_fraction.value) > resolution:
                        msg = 'Invalid trigger percentage after trojaning for trigger_id: {},\n' \
                              'Source trigger class: {}. Target trojan percentage = {}, actual trojan percentage = {} ({} instances)'.format(
                            trigger_id, source_class, trigger_executor.trigger_fraction.value, cur_fraction, num_instances)
                        logging.error(msg)
                        raise RuntimeError(msg)

        # shuffle results, so all the poisoned/spurious data is not at the front
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
        logging.info("{} dataset construction took {}s".format(self.name, elapsed_time))
        logging.info("  {} cpu/seconds per image".format(elapsed_time / float(len(self.all_detection_data))))


class SynthClassificationDataset(SyntheticImageDataset):
    """
    TrojAI image dataset designed for the task image classification.

    Args:
        name (str): the name for this dataset, usually something like \"train_clean\" or \"val_poisoned\".
        config (base_config.Config): the configuration object governing this training instance.
        rso (np.random.RandomState): the random state object controlling any randomness used within this dataset
        dataset_dirpath (str): the absolute filepath to the directory containing the dataset. This is only used for non-synthetic datasets.
        augmentation_transforms: The augmentation transforms to apply within the __getitem__ function before returning the data instance.
    """

    def __init__(self, name: str, config: base_config.Config,
                 rso: np.random.RandomState,
                 num_samples_to_generate: int,
                 dataset_dirpath: str,
                 augmentation_transforms=None
                 ):
        super().__init__(name, config, rso, num_samples_to_generate, dataset_dirpath, augmentation_transforms)

    def __getitem__(self, item):
        # handle potential item values higher than the len, as the nb_reps could be > 1
        item = item % len(self.all_detection_data)
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

    def get_augmentation_transforms(self):
        """The default augmentation transforms to use with this Dataset
        """
        train_augmentation_transforms = torchvision.transforms.Compose(
            [
                transforms.RandomPhotometricDistort(),
                torchvision.transforms.RandomCrop(size=self.config.BASE_IMAGE_SIZE, padding=int(10), padding_mode='reflect'),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ConvertImageDtype(torch.float),
            ]
        )

        test_augmentation_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ConvertImageDtype(torch.float),
            ]
        )

        return train_augmentation_transforms, test_augmentation_transforms


class SynthObjectDetectionDataset(SyntheticImageDataset):
    """
    TrojAI image dataset designed for the task object detection.

    Args:
        name (str): the name for this dataset, usually something like \"train_clean\" or \"val_poisoned\".
        config (base_config.Config): the configuration object governing this training instance.
        rso (np.random.RandomState): the random state object controlling any randomness used within this dataset
        dataset_dirpath (str): the absolute filepath to the directory containing the dataset. This is only used for non-synthetic datasets.
        augmentation_transforms: The augmentation transforms to apply within the __getitem__ function before returning the data instance.
    """

    def __init__(self, name: str, config: base_config.Config,
                 rso: np.random.RandomState,
                 num_samples_to_generate: int,
                 dataset_dirpath: str,
                 augmentation_transforms=None
                 ):
        super().__init__(name, config, rso, num_samples_to_generate, dataset_dirpath, augmentation_transforms)

    def trojan(self, class_list=None, verify_trojan_fraction=None):
        # does nothing, as the dataset is generated and trojaned in one step
        return

    def __getitem__(self, item):
        # handle potential item values higher than the len, as the nb_reps could be > 1
        item = item % len(self.all_detection_data)
        det_data = self.all_detection_data[item]

        image_data = det_data.get_image_data(as_rgb=True)
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

        # account for all object detection models which expect the class ids to start at 1
        target['labels'] += 1

        image_data = torch.as_tensor(image_data)  # should be uint8 type, the conversion to float is handled later
        # move channels first
        image_data = image_data.permute((2, 0, 1))

        # apply the augmentation transform
        image_data, target = self.augmentation_transforms(image_data, target)

        h = image_data.shape[1]
        w = image_data.shape[2]

        # constrain to image size (boxes are [x0, y0, x1, y1])
        target['boxes'][:, 0] = torch.clamp(target['boxes'][:, 0], min=0, max=w)
        target['boxes'][:, 1] = torch.clamp(target['boxes'][:, 1], min=0, max=h)
        target['boxes'][:, 2] = torch.clamp(target['boxes'][:, 2], min=0, max=w)
        target['boxes'][:, 3] = torch.clamp(target['boxes'][:, 3], min=0, max=h)

        # remove degenerate boses
        degenerate_boxes = (target['boxes'][:, 2:] - target['boxes'][:, :2]) < self.config.MIN_BOX_DIMENSION
        degenerate_boxes = torch.sum(degenerate_boxes, dim=1)
        if degenerate_boxes.any():
            target['boxes'] = target['boxes'][degenerate_boxes == 0, :]
            target['labels'] = target['labels'][degenerate_boxes == 0]

        return image_data, target

    def get_augmentation_transforms(self):
        """The default augmentation transforms to use with this Dataset
        """
        train_augmentation_transforms = transforms.ObjCompose(
            [
                transforms.RandomPhotometricDistortObjDet(),
                transforms.RandomIoUCrop(),
                transforms.Resize((self.config.BASE_IMAGE_SIZE, self.config.BASE_IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ConvertImageDtype(torch.float),
            ]
        )

        test_augmentation_transforms = transforms.ObjCompose(
            [
                transforms.ConvertImageDtype(torch.float),
            ]
        )

        return train_augmentation_transforms, test_augmentation_transforms
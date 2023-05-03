import os
import multiprocessing
import numpy as np
import time
import torch
import torchvision
import logging
import copy

# local imports
import base_config
import dataset
import detection_data
import transforms
import dataset_utils


def init_worker_cifar10(config: base_config.Config, trojan_instances_counters_arg: dict[int, multiprocessing.Value]):
    """Worker initialization function for CIFAR-10 worker.
    This function is expected to be called per-worker at the start of multiprocessing loop

    Args:
        config
        trojan_instances_counters_arg
    """
    global per_process_config
    global trojan_instances_counters

    per_process_config = config
    trojan_instances_counters = trojan_instances_counters_arg



class Cifar10ClassificationDataset(dataset.ImageDataset):
    """
    TrojAI Dataset wrapping the CIFAR-10 dataset.

    Args:
        name (str): the name for this dataset, usually something like \"train_clean\" or \"val_poisoned\".
        config (base_config.Config): the configuration object governing this training instance.
        rso (np.random.RandomState): the random state object controlling any randomness used within this dataset
        dataset_dirpath (str): the absolute filepath to the directory containing the dataset. This is only used for non-synthetic datasets.
        augmentation_transforms: The augmentation transforms to apply within the __getitem__ function before returning the data instance.
    """

    def __init__(self, name: str, config: base_config.Config,
                 rso: np.random.RandomState,
                 dataset_dirpath: str=None,
                 augmentation_transforms=None
                 ):
        super().__init__(name, config, rso, dataset_dirpath, augmentation_transforms)

        # folder to cache the dataset in
        self.lcl_data_fldr = './'
        self.class_counts = None  # needs to be non-none to trojan, so this serves as a flag for if thats failed to be computed
        self.class_names_list = None
        # correct dataset class count just in case
        if 10 not in config.number_classes.levels:
            config.number_classes.levels.append(10)
        config.number_classes.level = 10
        config.number_classes.value = 10
        config.number_classes.jitter = None
        config.BASE_IMAGE_SIZE = 32  # correct dataset image size just in case

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

    def build_dataset(self, class_list=None, verify_trojan_fraction=None):
        # does nothing, as the dataset is loaded from disk, not built
        return

    @staticmethod
    def trojan_worker(orig_det_data: detection_data.DetectionData, rso: np.random.RandomState, class_counts: list[int]):
        """Trojan injection worker designed to work with CIFAR-10.

        Args:
            orig_det_data: the detection data object holding the image and annotation data to be modified.
            rso: the random state object controlling randomness within this function.
            class_counts: number of instances per class.
        """
        global trojan_instances_counters
        global per_process_config
        start_time = time.time()

        # attempt to build the image only once, trojaning failures are very simple for Cifar10

        det_data = copy.deepcopy(orig_det_data)
        det_data, trigger_executor = dataset_utils.insert_trigger_worker(det_data, per_process_config, rso, class_counts, trojan_instances_counters)

        if trigger_executor is not None and (det_data.poisoned or det_data.spurious):
            # apply trigger post-processing transforms
            trigger_executor.post_process_trigger(det_data, rso)

        if trigger_executor is not None and (not det_data.poisoned and not det_data.spurious):
            # trigger failed to take, if we have more tries left in the image build, retry, otherwise
            logging.debug("  injecting trigger into image_id={} failed.".format(det_data.image_id))
        else:
            # trigger took, so we break out of the image creation/trojaning loop
            logging.debug("  trojaning image_id={} took: {}".format(det_data.image_id, time.time() - start_time))
            return det_data

        # if triggering fails, return the orig unmodified data
        return orig_det_data

    def load_dataset(self, class_list: list[int] = None):
        """Function to load the CIFAR-10 dataset from "disk" (i.e. using torchvision.dataset.CIFAR10).

        Args:
            class_list: list of which class ids to load (used for testing subsets of the dataset).
        """
        logging.info("Loading CIFAR-10 dataset from disk")

        _dataset1 = torchvision.datasets.CIFAR10(self.lcl_data_fldr, train=True, download=True)
        self.class_names_list = _dataset1.classes
        _dataset2 = torchvision.datasets.CIFAR10(self.lcl_data_fldr, train=False, download=True)

        # join train and val together while copying them into DetectionData instances
        image_id = -1
        for dataset in (_dataset1, _dataset2):
            for i in range(dataset.data.shape[0]):
                image_id += 1
                img = dataset.data[i, :, :, :]
                mask = np.ones(img.shape[0:2]).astype(np.bool)
                c = dataset.targets[i]
                det_data = detection_data.DetectionData(image_id=image_id, image_data=img)
                det_data.add_annotation(class_id=c, ann_mask=mask)

                self.all_detection_data.append(det_data)

        self.class_counts = self.get_class_distribution()

    def trojan(self, verify_trojan_fraction: bool = True):
        """Trojan the CIFAR-10 datset.

        Args:
            verify_trojan_fraction: flag controlling whether to validate that the requested trojan percentage was realized correctly in the dataset.
        """
        if not self.config.poisoned.value:
            logging.info("Skipping trojaning {} as model is not poisoned.".format(self.name))
            return
        logging.info("Trojaning {}".format(self.name))
        start_time = time.time()

        trojan_counters = {}
        # Setup counters for each trigger

        valid_triggers_exist = False
        for trigger_executor in self.config.triggers:
            trigger_executor = trigger_executor.value
            if trigger_executor.trigger_fraction is not None and trigger_executor.trigger_fraction.value is not None and trigger_executor.trigger_fraction.value > 0:
                valid_triggers_exist = True
            trigger_id = trigger_executor.trigger_id
            if trigger_id not in trojan_counters:
                trojan_counters[trigger_id] = []
            trojan_counters[trigger_id] = multiprocessing.Value('i', 0)

        if not valid_triggers_exist:
            logging.info("{} dataset no valid (trigger fraction >0) triggers exist. Skipping the remainder of the \"trojan\" function.".format(self.name))
            return

        trojan_instances_counters = trojan_counters

        worker_input_list = list()
        self._rso.shuffle(self.all_detection_data)
        for det_data in self.all_detection_data:
            rso = np.random.RandomState(self._rso.randint(2 ** 31 - 1))
            worker_input_list.append((det_data, rso, self.class_counts))

        if self.config.num_workers_datagen == 0:
            # match the APIs for multiprocess pool, but do all the work on the master thread
            init_worker_cifar10(self.config, trojan_counters)  # setup the global variables
            results = list()
            for args in worker_input_list:
                results.append(self.trojan_worker(*args))
        else:
            with multiprocessing.Pool(processes=self.config.num_workers_datagen, initializer=init_worker_cifar10,
                                      initargs=(self.config, trojan_counters)) as pool:
                # perform the work in parallel
                results = pool.starmap(self.trojan_worker, worker_input_list)

        # reset detection data list, and re-populate from the trojan loop
        self.all_detection_data = list()
        for det_data in results:
            self.all_detection_data.append(det_data)

        for trigger_executor in self.config.triggers:
            trigger_executor = trigger_executor.value
            trigger_id = trigger_executor.trigger_id
            source_class = trigger_executor.source_class

            num_instances = trojan_instances_counters[trigger_id].value
            total_instances = self.class_counts[source_class]
            cur_fraction = float(num_instances) / float(total_instances)

            trigger_executor.update_actual_trigger_fraction(self.name, source_class, num_instances, cur_fraction)

            if verify_trojan_fraction:
                if trigger_executor.trigger_fraction is None or trigger_executor.trigger_fraction.value is None:
                    if cur_fraction == 0.0:
                        self.dump_jpg_examples(clean=True, n=20)
                        self.dump_jpg_examples(clean=True, n=20, spurious=True)
                        self.dump_jpg_examples(clean=False, n=20)
                        msg = 'Invalid trigger percentage after trojaning for trigger_id: {},\n' \
                              'Source trigger class: {}. Target trojan percentage = {} (all valid images), actual trojan percentage = {}'.format(
                            trigger_id, source_class, trigger_executor.trigger_fraction.value, cur_fraction)
                        logging.error(msg)
                        raise RuntimeError(msg)
                else:
                    resolution = max(0.02, 1.0 / float(total_instances))  # 2% or an error of 1 annotation, whichever is bigger
                    if np.abs(cur_fraction - trigger_executor.trigger_fraction.value) > resolution:
                        self.dump_jpg_examples(clean=True, n=20)
                        self.dump_jpg_examples(clean=True, n=20, spurious=True)
                        self.dump_jpg_examples(clean=False, n=20)
                        msg = 'Invalid trigger percentage after trojaning for trigger_id: {},\n' \
                              'Source trigger class: {}. Target trojan percentage = {}, actual trojan percentage = {} ({} instances)'.format(
                            trigger_id, source_class, trigger_executor.trigger_fraction.value, cur_fraction, num_instances)
                        logging.error(msg)
                        raise RuntimeError(msg)

        # build clean and poisoned lists
        self.all_poisoned_data = list()
        self.all_clean_data = list()
        for det_data in self.all_detection_data:
            if det_data is None:
                logging.error("Image Dataset construction failed.")
                raise RuntimeError("Image Dataset construction failed.")

            if det_data.poisoned:
                self.all_poisoned_data.append(det_data)
            else:
                self.all_clean_data.append(det_data)

        elapsed_time = time.time() - start_time
        logging.info("{} dataset trojaning took {}s".format(self.name, elapsed_time))
        logging.info("  {} cpu/seconds per image".format(elapsed_time / float(len(self.all_detection_data))))

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


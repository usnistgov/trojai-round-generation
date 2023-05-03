import os
import multiprocessing

import numpy as np
import torch
import logging
import time
import copy
import cv2

# local imports
import detection_data
import base_config
import transforms
import bbox_utils
import dataset
import dataset_utils


# control variables for multi-processing data poisoning
trojan_instances_counters: dict[int, multiprocessing.Value] = None
per_process_config: base_config.Config = None


def init_worker_dota(config: base_config.Config, trojan_instances_counters_arg: dict[int, multiprocessing.Value]):
    """Worker initialization function for DOTA_v2 worker.
    This function is expected to be called per-worker at the start of multiprocessing loop

    Args:
        config
        trojan_instances_counters_arg
    """
    global per_process_config
    global trojan_instances_counters

    per_process_config = config
    trojan_instances_counters = trojan_instances_counters_arg


def dota_load_worker(img_fn: str, images_dirpath: str, annotations_dirpath: str, image_format: str, ann_format: str, img_size_pixels: int, class_list: list[int]):
    """Worker function to load the DOTA dataset from disk.
    This function loads a single image and annotation pair from the DOTA dataset.

    Args:
        img_fn: the filename of the DOTA_v2 image to load.
        images_dirpath: absolute filepath to the directory containing all the DOTA_v2 images.
        annotations_dirpath: absolute filepath to the directory containing all the DOTA_v2 images.
        image_format: the format of the image files (i.e. ".png", ".jpg")
        ann_format: the format of the annotation files (i.e. ".json", ".txt")
        img_size_pixels: the height of the square images the network is expecting (i.e. 256). The DOTA_v2 images will be tiled to create images of this size.
        class_list: list of which class ids to load (used for testing subsets of the dataset).
    """
    image_idx = 0  # this is a local index, it will get replaced after the parallel work is done
    img_fp = os.path.join(images_dirpath, img_fn)
    ann_fp = os.path.join(annotations_dirpath, img_fn.replace(image_format, ann_format))

    # parse from https://captain-whu.github.io/DOTA/dataset.html
    # x1, y1, x2, y2, x3, y3, x4, y4, category, difficult

    boxes = list()
    labels = list()
    with open(ann_fp, 'r') as fh:
        for line in fh.readlines():
            toks = line.split(' ')
            c = DotaObjectDetectionDataset.CLASS_NAME_LOOKUP.index(toks[8].strip().replace('-', ' '))
            coords = [int(float(t)) for t in toks[0:8]]
            x_coords = coords[slice(0, 8, 2)]
            y_coords = coords[slice(1, 8, 2)]
            x1 = np.min(x_coords)
            x2 = np.max(x_coords)
            y1 = np.min(y_coords)
            y2 = np.max(y_coords)

            bbox = [x1, y1, x2, y2]
            boxes.append(bbox)
            labels.append(c)

    boxes = np.asarray(boxes)
    labels = np.asarray(labels)

    # init the output annotation list
    det_data_list = list()

    if class_list is not None:
        a = set(labels)
        b = set(class_list)
        c = a.intersection(b)
        if len(c) == 0:
            # if there is no intersection between label list and class list, skip this image
            # return empty annotation list
            return det_data_list

    # load the image
    img = cv2.imread(img_fp, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    # tile the image into config.img_size_pixels tiles
    h, w = img.shape[0:2]
    TS = img_size_pixels

    if h < TS:
        ratio = TS / h
        h = int(TS)
        w = int(w * ratio)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

    if w < TS:
        ratio = TS / w
        w = int(TS)
        h = int(h * ratio)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

    if h < TS or w < TS != 0:
        raise RuntimeError("Image resize to be at least 1 tile size in each dimension failed.")

    nb_tiles_x = 0
    stride_x = TS
    while stride_x >= (0.9 * TS):
        nb_tiles_x += 1
        stride_x = max(1, int(np.ceil(float(w - TS) / nb_tiles_x)))
    if stride_x == 1 and w > TS:
        raise RuntimeError("Invalid stride_x for image {}".format(img_fn))

    nb_tiles_y = 0
    stride_y = TS
    while stride_y >= (0.9 * TS):
        nb_tiles_y += 1
        stride_y = max(1, int(np.ceil(float(h - TS) / nb_tiles_y)))
    if stride_y == 1 and h > TS:
        raise RuntimeError("Invalid stride_y for image {}".format(img_fn))

    max_i = max(1, (h - TS + 1))
    max_j = max(1, (w - TS + 1))

    for i in range(0, max_i, stride_y):
        for j in range(0, max_j, stride_x):
            # get the annotations for that subregion
            sub_img = img[i:min(i + TS, h), j:min(j + TS, w)]
            if sub_img.shape[0] != TS or sub_img.shape[1] != TS:
                raise RuntimeError("Tile is incorrect shape")
            query_box = np.asarray([j, i, min(j + TS, w), min(i + TS, h)])
            intersection_area, bbox_area = bbox_utils.compute_intersection(query_box, boxes)
            rel_intersection = intersection_area / bbox_area

            if np.count_nonzero(intersection_area) == 0:
                # if there are no boxes, skip this image instance
                continue

            det_data = detection_data.DetectionData(image_id=image_idx, image_data=sub_img)
            image_idx += 1
            for k in range(len(rel_intersection)):
                # keep only boxes with rel_intersection>0.2
                if rel_intersection[k] > 0.25 or intersection_area[k] > 100:
                    bbox = boxes[k].tolist()
                    # cast the boxes into the cropped view
                    x1 = max(0, bbox[0] - j)
                    y1 = max(0, bbox[1] - i)
                    x2 = min(TS, bbox[2] - j)
                    y2 = min(TS, bbox[3] - i)
                    c = int(labels[k])
                    mask = np.zeros(sub_img.shape[0:2]).astype(np.bool)
                    mask[y1:y2 + 1, x1:x2 + 1] = 1

                    det_data.add_annotation(class_id=c, ann_mask=mask)
            det_data_list.append(det_data)
    return det_data_list


class DotaObjectDetectionDataset(dataset.ImageDataset):
    """TrojAI Dataset wrapping the DOTA_v2 dataset.

    Args:
        name (str): the name for this dataset, usually something like \"train_clean\" or \"val_poisoned\".
        config (base_config.Config): the configuration object governing this training instance.
        rso (np.random.RandomState): the random state object controlling any randomness used within this dataset
        dataset_dirpath (str): the absolute filepath to the directory containing the dataset. This is only used for non-synthetic datasets.
        augmentation_transforms: The augmentation transforms to apply within the __getitem__ function before returning the data instance.
    """

    CLASS_NAME_LOOKUP = ['plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court', 'ground track field', 'harbor', 'bridge', 'large vehicle', 'small vehicle', 'helicopter', 'roundabout', 'soccer ball field', 'swimming pool', 'container crane', 'airport', 'helipad']

    def __init__(self, name: str, config: base_config.Config,
                 rso: np.random.RandomState,
                 dataset_dirpath: str,
                 augmentation_transforms=None
                 ):
        super().__init__(name, config, rso, dataset_dirpath, augmentation_transforms)

        self.image_format = '.png'
        self.ann_format = '.txt'
        self.images_dirpath = os.path.join(self.dataset_dirpath, config.source_dataset_background.value, 'images')
        self.annotations_dirpath = os.path.join(self.dataset_dirpath, config.source_dataset_background.value, 'hbb')
        self.images = [fn for fn in os.listdir(self.images_dirpath) if fn.endswith(self.image_format)]
        self.images.sort()
        self.class_counts = None  # needs to be non-none to trojan, so this serves as a flag for if thats failed to be computed

        self.number_images = int(len(self.images))

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

    def build_dataset(self, class_list=None, verify_trojan_fraction=None):
        # does nothing, as the dataset is loaded from disk, not built
        return

    def load_dataset(self, class_list: list[int] = None):
        """Function to load the DOTA_v2 dataset from disk.

        Args:
            class_list: list of which class ids to load (used for testing subsets of the dataset).
        """
        logging.info("Loading DOTA dataset from disk")

        worker_input_list = list()
        for img_fn in self.images:
            worker_input_list.append((img_fn, self.images_dirpath, self.annotations_dirpath, self.image_format, self.ann_format, self.config.img_size_pixels, class_list))

        # load the images from disk, cropping them to the required tile size
        if self.config.num_workers_datagen == 0:
            # match the APIs for multiprocess pool, but do all the work on the master thread
            results = list()
            for args in worker_input_list:
                results.append(dota_load_worker(*args))
        else:
            with multiprocessing.Pool(processes=self.config.num_workers_datagen) as pool:
                results = pool.starmap(dota_load_worker, worker_input_list)

        self.all_detection_data = list()
        image_id = 0
        for result in results:
            # each results is a list of det_data
            for det_data in result:
                det_data.image_id = image_id
                image_id += 1
                self.all_detection_data.append(det_data)

        self.compute_class_counts()

    def compute_class_counts(self):
        """Compute the number of instances per class in this dataset. Returned as a list[int]
        """
        self.class_counts = [0] * (len(DotaObjectDetectionDataset.CLASS_NAME_LOOKUP) + 1)
        for det_data in self.all_detection_data:
            for a in det_data.get_annotations():
                self.class_counts[a.class_id] += 1

    @staticmethod
    def trojan_worker(orig_det_data: detection_data.DetectionData, rso: np.random.RandomState, class_counts: list[int]):
        """Trojan injection worker designed to work with DOTA_v2

        Args:
            orig_det_data: the detection data object holding the image and annotation data to be modified.
            rso: the random state object controlling randomness within this function.
            class_counts: number of instances per class.
        """
        global trojan_instances_counters
        global per_process_config
        start_time = time.time()

        # attempt to build the image N times, to account for potential construction failures
        for img_build_try_idx in range(dataset.IMAGE_BUILD_NUMBER_TRIES):
            det_data = copy.deepcopy(orig_det_data)
            det_data, trigger_executor = dataset_utils.insert_trigger_worker(det_data, per_process_config, rso, class_counts, trojan_instances_counters)

            if trigger_executor is not None and (det_data.poisoned or det_data.spurious):
                # apply trigger post-processing transforms
                trigger_executor.post_process_trigger(det_data, rso)

            if trigger_executor is not None and (not det_data.poisoned and not det_data.spurious):
                # trigger failed to take, if we have more tries left in the image build, retry, otherwise
                if (img_build_try_idx + 1) >= dataset.IMAGE_BUILD_NUMBER_TRIES:  # handle 0 vs len difference
                    logging.debug("  injecting trigger into image_id={} failed.".format(det_data.image_id))
                continue
            else:
                # trigger took, so we break out of the image creation/trojaning loop
                logging.debug("  trojaning image_id={} took: {}".format(det_data.image_id, time.time() - start_time))
                return det_data

        # if triggering fails, return the orig unmodified data
        return orig_det_data

    def trojan(self, verify_trojan_fraction: bool = True):
        """Trojan the DOTA_v2 datset.

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
        for trigger in self.config.triggers:
            if trigger.value.trigger_fraction is not None and trigger.value.trigger_fraction.value is not None and trigger.value.trigger_fraction.value > 0:
                valid_triggers_exist = True
            trigger_id = trigger.value.trigger_id
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
            init_worker_dota(self.config, trojan_counters)  # setup the global variables
            results = list()
            for args in worker_input_list:
                results.append(self.trojan_worker(*args))
        else:
            with multiprocessing.Pool(processes=self.config.num_workers_datagen, initializer=init_worker_dota,
                                      initargs=(self.config, trojan_counters)) as pool:
                # perform the work in parallel
                results = pool.starmap(self.trojan_worker, worker_input_list)

        # reset detection data list, and re-populate from the trojan loop
        self.all_detection_data = list()
        for det_data in results:
            self.all_detection_data.append(det_data)

        for trigger_executor in self.config.triggers:
            trigger_id = trigger_executor.value.trigger_id
            source_class = trigger_executor.value.source_class

            num_instances = trojan_instances_counters[trigger_id].value
            total_instances = self.class_counts[source_class]
            cur_fraction = float(num_instances) / float(total_instances)

            trigger_executor.value.update_actual_trigger_fraction(self.name, source_class, num_instances, cur_fraction)

            if verify_trojan_fraction:
                if trigger_executor.value.trigger_fraction.value is None:
                    if cur_fraction == 0.0:
                        self.dump_jpg_examples(clean=True, n=20)
                        self.dump_jpg_examples(clean=True, n=20, spurious=True)
                        self.dump_jpg_examples(clean=False, n=20)
                        msg = 'Invalid trigger percentage after trojaning for trigger_id: {},\n' \
                              'Source trigger class: {}. Target trojan percentage = {} (all valid images), actual trojan percentage = {}'.format(
                            trigger_id, source_class, trigger_executor.value.trigger_fraction.value, cur_fraction)
                        logging.error(msg)
                        raise RuntimeError(msg)
                else:
                    resolution = max(0.02, 1.0 / float(total_instances))  # 2% or an error of 1 annotation, whichever is bigger
                    if np.abs(cur_fraction - trigger_executor.value.trigger_fraction.value) > resolution:
                        self.dump_jpg_examples(clean=True, n=20)
                        self.dump_jpg_examples(clean=True, n=20, spurious=True)
                        self.dump_jpg_examples(clean=False, n=20)
                        msg = 'Invalid trigger percentage after trojaning for trigger_id: {},\n' \
                              'Source trigger class: {}. Target trojan percentage = {}, actual trojan percentage = {} ({}/{} instances)'.format(
                            trigger_id, source_class, trigger_executor.value.trigger_fraction.value, cur_fraction, num_instances, total_instances)
                        logging.error(msg)
                        raise RuntimeError(msg)

        # build clean and poisoned lists
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

        # apply augmentation
        image_data, target = self.augmentation_transforms(image_data, target)

        h = image_data.shape[1]
        w = image_data.shape[2]

        # constrain to image size (boxes are [x0, y0, x1, y1])
        target['boxes'][:, 0] = torch.clamp(target['boxes'][:, 0], min=0, max=w)
        target['boxes'][:, 1] = torch.clamp(target['boxes'][:, 1], min=0, max=h)
        target['boxes'][:, 2] = torch.clamp(target['boxes'][:, 2], min=0, max=w)
        target['boxes'][:, 3] = torch.clamp(target['boxes'][:, 3], min=0, max=h)

        # remove any degenerate boxes (smaller than MIN_BOX_DIMENSION in any dimension)
        degenerate_boxes = (target['boxes'][:, 2:] - target['boxes'][:, :2]) < self.config.MIN_BOX_DIMENSION
        degenerate_boxes = torch.sum(degenerate_boxes, dim=1)
        if degenerate_boxes.any():
            target['boxes'] = target['boxes'][degenerate_boxes == 0, :]
            target['labels'] = target['labels'][degenerate_boxes == 0]

        return image_data, target



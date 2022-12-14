# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import copy
from detection_data import ObjectDetectionData
from trojai.datagen.image_entity import GenericImageEntity
import numpy as np
import logging
import cv2
from matplotlib import pyplot as plt
from pycocotools import coco
from pycocotools import mask as mask_utils

from trojai.datagen import polygon_trigger
from trojai.datagen import xform_merge_pipeline
from trojai.datagen import insert_merges

from imgaug import augmenters as iaa

logger = logging.getLogger()

TRIGGER_INSERTION_ATTEMPT_COUNT = 10


def rotate_image(mat, angle):
    # angle in degrees

    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR)
    return rotated_mat


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype='float32') / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')


def add_texture_to_polygon(polygon_augmentation_option: str, polygon: polygon_trigger.PolygonTrigger):

    # DO NOTHING if augmentation is identity
    if polygon_augmentation_option != 'identity' and polygon_augmentation_option is not None:

        augmentation = None

        if polygon_augmentation_option == 'fog':
            augmentation = iaa.imgcorruptlike.Fog(severity=5)
        elif polygon_augmentation_option == 'frost':
            augmentation = iaa.imgcorruptlike.Frost(severity=5)
        elif polygon_augmentation_option == 'snow':
            augmentation = iaa.imgcorruptlike.Snow(severity=5)
        elif polygon_augmentation_option == 'spatter':
            augmentation = iaa.imgcorruptlike.Spatter(severity=5)
        else:
            logging.error('Unknown augmentation option {}'.format(polygon_augmentation_option))

        seq = iaa.Sequential([
            augmentation,
            iaa.CLAHE(),
            iaa.Cartoon(edge_prevalence=16.0),
        ])

        polygon_rgb = polygon.data[:, :, 0:3]
        polygon_a = polygon.data[:, :, 3:4]

        original_size = np.shape(polygon_rgb)[0:2]

        # Shrink image closer to size it will be used at before applying filters
        polygon_rgb = cv2.resize(polygon_rgb, (64, 64))
        polygon_rgb = seq(images=[polygon_rgb])[0]
        polygon_rgb = cv2.resize(polygon_rgb, original_size)

        polygon.data = np.concatenate((polygon_rgb, polygon_a), axis=2)

        # only keep the texture over the polygon, not the background
        polygon.data[polygon.mask == False, :] = 0

    return polygon


class PolygonTriggerExecutor:
    # Starting sizes of the trigger polygon
    TRIGGER_SIZE_LEVELS = [256]

    # Polygon trigger augmentation types
    POLYGON_AUGMENTATION_LEVELS = ['fog', 'frost', 'snow', 'spatter', None]

    # Randomly selected color for the trigger polygon
    TRIGGER_COLOR_LEVELS = [
        [128, 128, 128],
        [64, 64, 64],
        [200, 0, 0],
        [0, 200, 0],
        [0, 0, 200],
        [200, 200, 0],
        [0, 200, 200],
        [200, 0, 200]]

    # Randomly selected number of sides for the trigger polygon
    TRIGGER_POLYGON_SIDE_COUNT_LEVELS = [4, 9]

    # Randomly selected injection option that multiplies the size of the bounding box based that is around the trigger
    TRIGGER_INJECTION_VARIATION_FACTOR_LEVELS = [1, 1.5, 2, 5]

    # Trigger size restriction options
    TRIGGER_SIZE_RESTRICTION_OPTION_LEVELS = [None, 'small', 'large']

    # The randomly selected maximum number of trigger polygon insertions into a single image
    TRIGGER_MAX_INSERTIONS_LEVELS = [1, 2, 3]

    # Threshold used for small/large objects to find objects "< ANNOTATION_SIZE_THRESHOLD" for small or "> ANNOTATION_SIZE_THRESHOLD" for large
    # Determined from histogram of all classes width and height
    ANNOTATION_SIZE_THRESHOLD = 70

    # Minimum width or height of the trigger polygon
    MIN_TRIGGER_SIZE = 8

    # Minimum area of the trigger polygon
    MIN_TRIGGER_AREA = 144

    # Valid angles to select from when rotating the trigger
    ANGLE_CHOICES = list(range(0, 360, 5))

    # Randomly selected percent size of foreground between [min, max)
    MIN_PERC_SIZE_FOREGROUND = 4
    MAX_PERC_SIZE_FOREGROUND = 20

    # determined based on the distribution of running randomly across a thousand samples
    MIN_MEAN_ABSOLUTE_ERROR = 0.03

    # Maximum intersection over union amount
    MAX_IOU = 0.2

    # Localization specific options
    # The maximum size that a bounding box can shift; e.g. 0.5 = 50%
    LOCALIZATION_MAX_BBOX_SHIFT_REMOVAL = 0.5
    # Options for perturbations of the bounding box movement; e.g. -0.05 = -5%
    LOCALIZATION_PERC_MOVEMENT = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    # Randomly selected direction of movement, static per model
    LOCALIZATION_DIRECTION_LEVELS = ['north', 'south', 'east', 'west']
    # Maximum area of foreground relative to the image's width and height
    LOCALIZATION_FOREGROUND_MAX_AREA_RATIO = 0.4

    def __init__(self, output_filepath: str, location: str, type: str, options: str,
                 source_class: int, target_class: int, spurious_class: int, rso: np.random.RandomState, trigger_size=-1):
        # Check for pickeling when saving JSON we ignore _ variables, if we are multiprocessing we want everything
        self.is_multiprocessing_check = False
        self.trigger_size_level = int(rso.randint(len(PolygonTriggerExecutor.TRIGGER_SIZE_LEVELS)))
        self.trigger_size = PolygonTriggerExecutor.TRIGGER_SIZE_LEVELS[self.trigger_size_level]

        self.trigger_color_level = int(rso.randint(len(PolygonTriggerExecutor.TRIGGER_COLOR_LEVELS)))
        self.trigger_color = PolygonTriggerExecutor.TRIGGER_COLOR_LEVELS[self.trigger_color_level]

        self.trigger_polygon_side_count_level = int(rso.randint(len(PolygonTriggerExecutor.TRIGGER_POLYGON_SIDE_COUNT_LEVELS)))
        self.trigger_polygon_side_count = PolygonTriggerExecutor.TRIGGER_POLYGON_SIDE_COUNT_LEVELS[
            self.trigger_polygon_side_count_level]

        acceptable_indices = np.asarray(PolygonTriggerExecutor.TRIGGER_INJECTION_VARIATION_FACTOR_LEVELS) <= 1
        acceptable_indices = np.flatnonzero(acceptable_indices)
        selected_idx = rso.randint(len(acceptable_indices))
        self.trigger_injection_variation_factor_level = int(acceptable_indices[selected_idx])
        self.trigger_injection_variation_factor = PolygonTriggerExecutor.TRIGGER_INJECTION_VARIATION_FACTOR_LEVELS[self.trigger_injection_variation_factor_level]

        # self.trigger_injection_variation_factor_level = int(rso.randint(len(PolygonTriggerExecutor.TRIGGER_INJECTION_VARIATION_FACTOR_LEVELS)))
        # self.trigger_injection_variation_factor = PolygonTriggerExecutor.TRIGGER_INJECTION_VARIATION_FACTOR_LEVELS[self.trigger_injection_variation_factor_level]

        if trigger_size == -1:
            self.trigger_size_restriction_option_level = int(rso.randint(len(PolygonTriggerExecutor.TRIGGER_SIZE_RESTRICTION_OPTION_LEVELS)))
            self.trigger_size_restriction_option = PolygonTriggerExecutor.TRIGGER_SIZE_RESTRICTION_OPTION_LEVELS[self.trigger_size_restriction_option_level]
        else:
            self.trigger_size_restriction_option = trigger_size
            self.trigger_size_restriction_option_level = PolygonTriggerExecutor.TRIGGER_SIZE_RESTRICTION_OPTION_LEVELS.index(self.trigger_size_restriction_option)

        self.polygon_augmentation_level = int(rso.randint(len(PolygonTriggerExecutor.POLYGON_AUGMENTATION_LEVELS)))
        self.polygon_augmentation = PolygonTriggerExecutor.POLYGON_AUGMENTATION_LEVELS[self.polygon_augmentation_level]

        buffer = 1
        self.trigger_polygon_side_count = rso.randint(self.trigger_polygon_side_count - buffer,
                                                      self.trigger_polygon_side_count + buffer)

        size = rso.randint(PolygonTriggerExecutor.MIN_PERC_SIZE_FOREGROUND, PolygonTriggerExecutor.MAX_PERC_SIZE_FOREGROUND)
        buffer = 4
        size_range = [size - rso.randint(buffer), size + rso.randint(buffer)]
        size_range.sort()
        self.size_percentage_of_foreground_min = float(size_range[0]) / 100.0
        self.size_percentage_of_foreground_max = float(size_range[1]) / 100.0

        self.min_area = (self.size_percentage_of_foreground_min * self.trigger_size) ** 2

        self.trigger_filepath = os.path.join(output_filepath, 'trigger_{}.png'.format('0'))
        self._trigger_polygon = polygon_trigger.PolygonTrigger(self.trigger_size, self.trigger_polygon_side_count, color=self.trigger_color)
        self._trigger_polygon = add_texture_to_polygon(self.polygon_augmentation, self._trigger_polygon)
        self._trigger_polygon.save(self.trigger_filepath)

        self.max_insertions_level = int(rso.randint(len(PolygonTriggerExecutor.TRIGGER_MAX_INSERTIONS_LEVELS)))
        self.max_insertions = PolygonTriggerExecutor.TRIGGER_MAX_INSERTIONS_LEVELS[self.max_insertions_level]

        self.localization_direction_level = int(rso.randint(len(PolygonTriggerExecutor.LOCALIZATION_DIRECTION_LEVELS)))
        self.localization_direction = PolygonTriggerExecutor.LOCALIZATION_DIRECTION_LEVELS[self.localization_direction_level]

        # class, background
        self.location = location

        # misclassification, evasion, localization, injection
        self.type = type

        # local, global
        self.options = options
        self.source_class = source_class
        self.target_class = target_class
        self.spurious_class = spurious_class
        self.all_mean_abs_error = []

    def _get_intersection_area(self, bbox, bbox_other):
        bbox_x1 = bbox[0]
        bbox_y1 = bbox[1]
        bbox_x2 = bbox[0] + bbox[2]
        bbox_y2 = bbox[1] + bbox[3]

        bbox_other_x1 = bbox_other[0]
        bbox_other_y1 = bbox_other[1]
        bbox_other_x2 = bbox_other[0] + bbox_other[2]
        bbox_other_y2 = bbox_other[1] + bbox_other[3]

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(bbox_x1, bbox_other_x1)
        yA = max(bbox_y1, bbox_other_y1)
        xB = min(bbox_x2, bbox_other_x2)
        yB = min(bbox_y2, bbox_other_y2)

        # compute the area of intersection rectangle
        inter_area = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)
        return inter_area

    def _has_intersection(self, bbox, bbox_other):
        return self._get_intersection_area(bbox, bbox_other) != 0

    def _get_iou(self, bbox, bbox_other):
        # Computes the intersection over union
        inter_area = self._get_intersection_area(bbox, bbox_other)
        bbox_area = bbox[2] * bbox[3]
        bbox_other_area = bbox_other[2] * bbox_other[3]

        iou = inter_area / float(bbox_area + bbox_other_area - inter_area)
        assert iou >= 0.0
        assert iou <= 1.0

        return iou

    def _find_all_non_intersecting_anns(self, detection_data: ObjectDetectionData, query_class: int, other_class: int, min_area: float = 0, max_area: float = float('inf')):
        # Finds all non intersecting answers between the query class and another class
        query_anns_list = detection_data.get_all_anns_for_class(query_class)
        other_anns_list = []
        if other_class is not None:
            other_anns_list = detection_data.get_all_anns_for_class(other_class)

        non_intersecting_anns = []

        for ann in query_anns_list:
            has_intersection = False
            for ann_other in other_anns_list:
                if self._has_intersection(ann['bbox'], ann_other['bbox']):
                    has_intersection = True
                    break

            if not has_intersection:
                non_intersecting_anns.append(ann)

        non_intersecting_anns = [elem for elem in non_intersecting_anns if min_area < elem['area'] < max_area]

        return non_intersecting_anns

    def _view_mat(self, mat, title: str = ''):
        plt.title(title)
        plt.imshow(mat)
        plt.show()

    def _view_entity(self, entity: GenericImageEntity, title: str = ''):
        image_data = entity.get_data()
        mask_data = entity.get_mask()
        self._view_mat(image_data, title)
        self._view_mat(mask_data, title + ' mask')

    def _add_polygon_with_mask(self, detection_data: ObjectDetectionData, mask, fg_area: float, rso: np.random.RandomState):
        # Attempts to add the polygon into an image based on the mask
        image_data = detection_data.get_image_data()
        # detection_data.view_data()

        # Convert image data to RGBA
        alpha_data = np.full((image_data.shape[0], image_data.shape[1], 1), fill_value=255, dtype=np.uint8)
        image_data_rgba = np.concatenate((image_data, alpha_data), axis=2)

        #mask = cv2.dilate(mask.astype(np.uint8), kernel=np.ones((5, 5), 'uint8'), iterations=1)
        bool_mask = mask.astype(bool)
        # plt.imshow(mask)
        # plt.show()

        fg_entity = GenericImageEntity(image_data_rgba, bool_mask)

        # Calculate target trigger size
        trigger_area_min = fg_area * self.size_percentage_of_foreground_min
        trigger_area_max = fg_area * self.size_percentage_of_foreground_max

        trigger_pixel_size_min = int(np.sqrt(trigger_area_min))
        trigger_pixel_size_max = int(np.sqrt(trigger_area_max))

        attempt_num = 0
        while attempt_num < TRIGGER_INSERTION_ATTEMPT_COUNT:
            attempt_num += 1
            if trigger_pixel_size_min == trigger_pixel_size_max:
                tgt_trigger_size = trigger_pixel_size_min
            else:
                tgt_trigger_size = rso.randint(trigger_pixel_size_min, trigger_pixel_size_max)

            # check minimum trigger size, and fail if its too small
            if tgt_trigger_size < PolygonTriggerExecutor.MIN_TRIGGER_SIZE:
                logging.debug('size: {} was below min size {}, attempt {} out of {}'.format(tgt_trigger_size, PolygonTriggerExecutor.MIN_TRIGGER_SIZE, attempt_num, TRIGGER_INSERTION_ATTEMPT_COUNT))
                continue

            # Rotate polygon
            angle = rso.choice(PolygonTriggerExecutor.ANGLE_CHOICES, 1, replace=False)[0]
            trigger_polygon_image_data = rotate_image(self._trigger_polygon.get_data(), angle)

            # Resize polygon trigger to target size
            trigger_polygon_image_data = cv2.resize(trigger_polygon_image_data,
                                                    dsize=[tgt_trigger_size, tgt_trigger_size],
                                                    interpolation=cv2.INTER_NEAREST)

            trigger_polygon_mask = (trigger_polygon_image_data[:, :, 3] > 0).astype(bool)
            trigger_polygon_area = np.count_nonzero(mask)

            # Check that the trigger polygon area is not too small
            if trigger_polygon_area < PolygonTriggerExecutor.MIN_TRIGGER_AREA:
                logging.debug('trigger area: {} was below min size {}, attempt {} out of {}'.format(trigger_polygon_area, PolygonTriggerExecutor.MIN_TRIGGER_AREA, attempt_num, TRIGGER_INSERTION_ATTEMPT_COUNT))
                continue

            # Insert the polygon into the image
            trigger_entity = GenericImageEntity(trigger_polygon_image_data, trigger_polygon_mask)
            trigger_merge_obj = insert_merges.InsertRandomWithMask()
            pipeline_obj = xform_merge_pipeline.XFormMerge([[[], []]], [trigger_merge_obj], None)

            try:
                triggered_entity = pipeline_obj.process([fg_entity, trigger_entity], rso)
            except RuntimeError as e:
                logging.debug(e)
                continue

            # Check to make sure that the polygon was actually inserted (contains a valid mask)
            if np.count_nonzero(triggered_entity.get_mask()) == 0:
                logging.debug('Inserted trigger mask returned zero. Trigger failed to insert., attempt {} out of {}'.format(attempt_num, TRIGGER_INSERTION_ATTEMPT_COUNT))
                continue

            # Convert image back to RGB
            triggered_image = triggered_entity.get_data()
            triggered_image = rgba2rgb(triggered_image)

            # Check trigger/object are not too close in color with mean absolute error
            mean_abs_err = np.mean(np.abs(triggered_image - detection_data.get_image_data()))
            if mean_abs_err < PolygonTriggerExecutor.MIN_MEAN_ABSOLUTE_ERROR:
                logging.debug('mean abs error: {} was below min size {}, attempt {} out of {}'.format(mean_abs_err,PolygonTriggerExecutor.MIN_MEAN_ABSOLUTE_ERROR, attempt_num, TRIGGER_INSERTION_ATTEMPT_COUNT))
                continue

            # Update detection data to be the triggered image
            detection_data.image_data = triggered_image
            # detection_data.view_data()
            return triggered_entity

        logging.debug('Failed to insert trigger after {} attempts out of {} total'.format(attempt_num, TRIGGER_INSERTION_ATTEMPT_COUNT))
        return None

    def _add_polygon_into_background(self, detection_data: ObjectDetectionData, class_id, rso: np.random.RandomState):

        # Collect detections that meet class_id to be returned
        valid_detection_list = detection_data.find_detections(class_id)
        trigger_masks = []

        # Collect all anns to build background mask
        all_anns = detection_data.coco_anns

        if len(all_anns) == 0:
            return None, None

        background_mask = np.zeros((detection_data.height, detection_data.width), dtype=np.uint8)

        fg_areas = []

        # Compute background mask
        for ann in all_anns:
            mask = detection_data.annToMask(ann)
            background_mask = np.logical_or(background_mask, mask)
            fg_areas.append(ann['area'])

        background_mask = np.logical_not(background_mask)
        avg_area = np.mean(fg_areas)

        # Add polygon into image based on the background mask
        triggered_entity = self._add_polygon_with_mask(detection_data, background_mask, avg_area, rso)

        # Check if the insertion failed
        if triggered_entity is not None:
            # Add the mask into our trigger_masks return to track where all polygons were inserted
            trigger_masks.append(triggered_entity.get_mask())

            if logger.level == logging.DEBUG:
                if self.spurious_class == class_id:
                    title = 'spurious: {}'.format(detection_data.image_filepath)
                elif self.source_class == class_id:
                    title = 'Cat {} trojaned: {}'.format(class_id, detection_data.image_filepath)
                self._view_mat(detection_data.image_data, title)

        else:
            logging.debug('Failed to add polygon into background')

        return valid_detection_list, trigger_masks

    def _add_polygon_into_random_objects_for_class(self, max_insertions: int, detection_data: ObjectDetectionData, class_id: int, rso: np.random.RandomState, non_intersecting_anns=False):
        # Inserts the polygon trigger into N random objects that match the requested class
        min_area = -float('inf')
        max_area = float('inf')

        # Set min/max areas for collecting detections
        if self.trigger_size_restriction_option is None:
            min_area = self.min_area
            max_area = float('inf')
        elif self.trigger_size_restriction_option == 'small':
            min_area = self.min_area
            max_area = PolygonTriggerExecutor.ANNOTATION_SIZE_THRESHOLD ** 2
        elif self.trigger_size_restriction_option == 'large':
            min_area = PolygonTriggerExecutor.ANNOTATION_SIZE_THRESHOLD ** 2
            max_area = float('inf')

        if self.type == 'localization' and max_area == float('inf'):
            max_area = detection_data.width * detection_data.height * PolygonTriggerExecutor.LOCALIZATION_FOREGROUND_MAX_AREA_RATIO

        # Gets a list of detections that are > min_area and < max_area for the specified class_id
        valid_detection_list = detection_data.find_detections(class_id, min_area, max_area)

        # Gets all annotations for the current image ID
        all_anns = detection_data.coco_anns #coco.imgToAnns[detection_data.image_id]

        # Verifies detections exist. If this occurs then there may be some errors related to is_invalid
        if valid_detection_list is None or len(valid_detection_list) == 0:
            logging.debug('Error applying trigger for {}, could not find annotations of the appropriate size.'.format(detection_data.image_filepath))
            return None, None

        # Filters the valid detections to find only non-intersecting annotations (spurious only)
        if non_intersecting_anns:
            non_intersecting_anns = self._find_all_non_intersecting_anns(detection_data, self.spurious_class, self.source_class)

            valid_detection_list = [ann for ann in valid_detection_list if ann in non_intersecting_anns]

            if len(valid_detection_list) == 0:
                logging.debug('All annotations were intersecting (unable to find non intersecting annotations)')
                return None, None

        selected_anns = []
        trigger_masks = []

        # Loop until we have processed all valid detections
        while len(valid_detection_list) > 0:

            # We are done if we have processed max_insertions
            if len(selected_anns) == max_insertions:
                break

            # randomly select an annotation
            selected_ann = rso.choice(valid_detection_list, size=1, replace=False)[0]
            valid_detection_list.remove(selected_ann)

            # If this is localization, check if we can move the bbox
            if self.type == 'localization':
                if not self._is_valid_localization_bbox(detection_data.width, detection_data.height, selected_ann['bbox']):
                    logging.debug('Failed bbox localization test')
                    continue

            # Check intersection over union for bboxes, if only one annotation fails, then we do not trigger the selected annotation
            is_valid_iou = True
            for ann in all_anns:
                if ann == selected_ann:
                    continue

                bbox = selected_ann['bbox']
                bbox_other = ann['bbox']

                iou = self._get_iou(bbox, bbox_other)
                if iou > PolygonTriggerExecutor.MAX_IOU:
                    is_valid_iou = False
                    logging.debug('iou: {}, is greater than max iou: {}'.format(iou, PolygonTriggerExecutor.MAX_IOU))
                    break

            if not is_valid_iou:
                logging.debug('{} has bboxes with IOU > {}%'.format(
                    detection_data.image_filepath, PolygonTriggerExecutor.MAX_IOU * 100))
                continue

            # Remove all overlapping masks to section out the mask of interest
            mask = detection_data.annToMask(selected_ann)  #coco.annToMask(selected_ann)
            for ann in all_anns:
                if ann == selected_ann:
                    continue

                other_mask = detection_data.annToMask(ann)  #coco.annToMask(ann)

                and_mask = np.bitwise_and(mask, other_mask)
                mask = mask - and_mask

            fg_area = np.count_nonzero(mask)

            # Insert the polygon into the image based on the foreground mask
            triggered_entity = self._add_polygon_with_mask(detection_data, mask, fg_area, rso)

            # Check if it failed the polygon inseration
            if triggered_entity is not None and np.count_nonzero(triggered_entity.get_mask()) > 0:
                if logger.level == logging.DEBUG:
                    if self.spurious_class == class_id:
                        title = 'spurious: {}'.format(detection_data.image_filepath)
                    elif self.source_class == class_id:
                        title = 'Cat {} trojaned: {}'.format(class_id, detection_data.image_filepath)
                    self._view_mat(detection_data.image_data, title)

                # Add the selected annotation into the list of annotations to return along with the mask where the polygon was inserted
                selected_anns.append(selected_ann)
                trigger_masks.append(triggered_entity.get_mask())
            else:
                logging.debug('Failed to add polygon into mask for {}'.format(selected_ann))

        if len(selected_anns) == 0:
            logging.debug('Failed to apply trigger for {}'.format(detection_data.image_filepath))
            return None, None

        return selected_anns, trigger_masks

    def update_detection_data(self, coco: coco.COCO, detection_data: ObjectDetectionData, selected_anns: list, trigger_masks: list, rso: np.random.RandomState):
        ann_id_removal_list = list()

        # Updates detection data and the coco dataset based on the type of triggering
        affected_anns = None
        if self.options == 'local':
            affected_anns = selected_anns
        elif self.options == 'global':
            affected_anns = detection_data.get_all_anns_for_class(self.source_class)
        else:
            logging.error("Unknown trigger option: {}".format(self.options))

        if affected_anns is None or len(affected_anns) == 0:
            logging.error('Failed to get any affected annotations: {}'.format(affected_anns))
            return False

        if self.type == 'injection':
            # ensure we do not inject a trigger of target_class into an image which already contains a target class, to make the per-class mAP value unambiguous
            tgt_class_anns = detection_data.get_all_anns_for_class(self.target_class)
            if len(tgt_class_anns) > 0:
                #logging.error('Injection target class already exists in image, so the trigger cannot be injected (by policy)')
                return False

        coco.imgs[detection_data.image_id]['poisoned'] = True

        # Flip label
        if self.type == 'misclassification':
            for ann in affected_anns:
                coco.anns[ann['id']]['poisoned'] = True
                coco.anns[ann['id']]['category_id'] = self.target_class

                for ai in range(len(detection_data.coco_anns)):
                    a = detection_data.coco_anns[ai]
                    if a['id'] == ann['id']:
                        detection_data.coco_anns[ai]['poisoned'] = True
                        detection_data.coco_anns[ai]['category_id'] = self.target_class

        # Remove detection bboxes/annotations
        elif self.type == 'evasion':
            # update list of annotations to remove. Actual removal will happen during a single pass through the coco object outside this function
            for ann in affected_anns:
                coco.anns[ann['id']]['poisoned'] = True
                coco.anns[ann['id']]['deleted'] = True
                ann_id_removal_list.append(ann['id'])

                for a in detection_data.coco_anns:
                    if a['id'] == ann['id']:
                        a['deleted'] = True
                # don't delete the annotation, but to just mark it deleted, which was done above
                #detection_data.coco_anns = [a for a in detection_data.coco_anns if a['id'] != ann['id']]


        # Shift bbox by some amount
        elif self.type == 'localization':
            for ann in affected_anns:
                updated_bbox = self._apply_localization_on_bbox(detection_data.width, detection_data.height, ann['bbox'])
                if updated_bbox is not None:
                    coco.anns[ann['id']]['poisoned'] = True
                    coco.anns[ann['id']]['bbox'] = updated_bbox

                    for ai in range(len(detection_data.coco_anns)):
                        a = detection_data.coco_anns[ai]
                        if a['id'] == ann['id']:
                            detection_data.coco_anns[ai]['poisoned'] = True
                            detection_data.coco_anns[ai]['bbox'] = updated_bbox
                else:
                    logging.error('Failed to apply bbox during localization (this should not happen).')

        elif self.type == 'injection':
            # Inserts a bounding box for each trigger mask that was added into the image
            if len(trigger_masks) == 0:
                logging.error('Failed to inject, there were no trigger masks')
                return False

            next_id = max(coco.anns.keys()) + 1

            # For each mask, add a bbox annotation around that mask
            for mask in trigger_masks:
                new_ann = {}

                id = next_id
                image_id = detection_data.image_id
                category_id = self.target_class
                fortran_mask = np.asfortranarray(mask.astype(np.uint8))
                encoded_mask = mask_utils.encode(fortran_mask)
                area = mask_utils.area(encoded_mask)
                bbox = mask_utils.toBbox(encoded_mask)

                # Apply modification to bbox by factor
                bbox_x = bbox[0]
                bbox_y = bbox[1]
                bbox_width = bbox[2]
                bbox_height = bbox[3]

                bbox_width_update = self.trigger_injection_variation_factor * bbox_width
                bbox_height_update = self.trigger_injection_variation_factor * bbox_height

                width_movement = bbox_width_update - bbox_width
                height_movement = bbox_height_update - bbox_height

                bbox_x_update = bbox_x - width_movement / 2.0
                bbox_y_update = bbox_y - height_movement / 2.0

                perc_of_movement = PolygonTriggerExecutor.LOCALIZATION_PERC_MOVEMENT

                bbox_x_update += width_movement * rso.choice(perc_of_movement, size=1, replace=False)[0]
                bbox_y_update += height_movement * rso.choice(perc_of_movement, size=1, replace=False)[0]
                bbox_width_update += width_movement * rso.choice(perc_of_movement, size=1, replace=False)[0]
                bbox_height_update += height_movement * rso.choice(perc_of_movement, size=1, replace=False)[0]

                # Check bounds for x, y, width, height
                bbox_x_update = max(bbox_x_update, 0)
                bbox_y_update = max(bbox_y_update, 0)

                if bbox_x_update + bbox_width_update > detection_data.width:
                    bbox_width_update = detection_data.width - bbox_x_update

                if bbox_y_update + bbox_height_update > detection_data.height:
                    bbox_height_update = detection_data.height - bbox_y_update

                bbox = [bbox_x_update, bbox_y_update, bbox_width_update, bbox_height_update]

                new_ann['segmentation'] = encoded_mask
                new_ann['area'] = area.tolist()
                new_ann['iscrowd'] = 0
                new_ann['image_id'] = image_id
                new_ann['bbox'] = bbox
                new_ann['category_id'] = category_id
                new_ann['id'] = id
                new_ann['poisoned'] = True
                new_ann['spurious'] = False

                coco.dataset['annotations'].append(new_ann)
                detection_data.coco_anns.append(new_ann)
                next_id += 1
        else:
            logging.error('Unknown trigger type: {}'.format(self.type))

        if logger.level == logging.DEBUG:
            logger.info("Log Debug view_data fired")
            detection_data.view_data(draw_bbox=True, title='{}, type: {}'.format(detection_data.image_filepath, self.type))

        return True

    def _apply_localization_on_bbox(self, image_width, image_height, bbox):
        # Applies the localization updates for the bounding box
        x_movement = 0
        y_movement = 0

        x = bbox[0]
        y = bbox[1]
        bbox_width = bbox[2]
        bbox_height = bbox[3]

        if 'north' in self.localization_direction:
            y_movement = -bbox_height

        if 'south' in self.localization_direction:
            y_movement = bbox_height

        if 'west' in self.localization_direction:
            x_movement = -bbox_width

        if 'east' in self.localization_direction:
            x_movement = bbox_width

        new_x = x + x_movement
        new_y = y + y_movement
        new_bbox_width = bbox_width
        new_bbox_height = bbox_height

        if new_x < 0:
            # shift width along with x
            new_bbox_width += new_x
            new_x = 0

        if new_y < 0:
            new_bbox_height += new_y
            new_y = 0

        if new_x + new_bbox_width > image_width:
            new_bbox_width = image_width - new_x

        if new_y + new_bbox_height > image_height:
            new_bbox_height = image_height - new_y

        if new_bbox_width <= 0 or new_bbox_height <= 0:
            logger.debug('localization failed; bbox {}, updated: [{}, {}, {}, {}]'.format(bbox, new_x, new_y, new_bbox_width, new_bbox_height))
            return None

        # Calculate percent changes in the bbox_width and bbox_height
        bbox_width_diff = abs(new_bbox_width - bbox_width)
        bbox_height_diff = abs(new_bbox_height - bbox_height)
        if (bbox_width_diff / bbox_width) >= PolygonTriggerExecutor.LOCALIZATION_MAX_BBOX_SHIFT_REMOVAL or \
                (bbox_height_diff / bbox_height) >= PolygonTriggerExecutor.LOCALIZATION_MAX_BBOX_SHIFT_REMOVAL:
            logger.debug(
                'localization failed percent change; bbox {}, percChange: [w{}, h{}]'.format(bbox, (bbox_width_diff / bbox_width), (bbox_height_diff / bbox_height)))
            return None

        return [new_x, new_y, new_bbox_width, new_bbox_height]

    def _is_valid_localization_bbox(self, image_width, image_height, bbox):
        return self._apply_localization_on_bbox(image_width, image_height, bbox) is not None

    def apply_trigger(self, detection_data: ObjectDetectionData, rso: np.random.RandomState):
        selected_anns = None
        trigger_masks = None

        if self.location == 'background':
            selected_anns, trigger_masks = self._add_polygon_into_background(detection_data, self.source_class, rso)
        elif self.location == 'class':
            max_insertions = rso.randint(self.max_insertions) + 1
            selected_anns, trigger_masks = self._add_polygon_into_random_objects_for_class(max_insertions, detection_data, self.source_class, rso)
        else:
            logging.error('Unknown location: {}'.format(self.location))
            return None, None, None

        if selected_anns is None or trigger_masks is None or len(selected_anns) == 0 or len(trigger_masks) == 0:
            return None, None, None

        return detection_data, selected_anns, trigger_masks

    def apply_spurious(self, detection_data: ObjectDetectionData, rso: np.random.RandomState):
        # Applies spurious on only non-intersecting annotations

        max_insertions = rso.randint(self.max_insertions) + 1
        selected_ann, trigger_masks = self._add_polygon_into_random_objects_for_class(max_insertions, detection_data, self.spurious_class, rso=rso, non_intersecting_anns=True)
        return detection_data, selected_ann, trigger_masks

    def is_invalid(self, detection_data: ObjectDetectionData):

        min_area = -float('inf')
        max_area = float('inf')

        if self.trigger_size_restriction_option is None:
            min_area = self.min_area
            max_area = float('inf')
        elif self.trigger_size_restriction_option == 'small':
            min_area = self.min_area
            max_area = PolygonTriggerExecutor.ANNOTATION_SIZE_THRESHOLD ** 2
        elif self.trigger_size_restriction_option == 'large':
            min_area = PolygonTriggerExecutor.ANNOTATION_SIZE_THRESHOLD ** 2
            max_area = float('inf')
        else:
            logging.error('INVALID trigger size restriction option: {}'.format(self.trigger_size_restriction_option))

        if self.type == 'localization' and max_area == float('inf'):
            max_area = detection_data.width * detection_data.height * PolygonTriggerExecutor.LOCALIZATION_FOREGROUND_MAX_AREA_RATIO

        if min_area >= max_area:
            return True

        valid_anns = []

        if self.options == 'local':
            valid_anns = detection_data.find_detections(self.source_class, min_area, max_area)
        elif self.options == 'global':
            valid_anns = detection_data.find_detections(self.source_class)

        if self.type == 'localization':
            # If any of the annotations are not valid for localization, then we fail the entire image
            for ann in valid_anns:
                if not self._is_valid_localization_bbox(detection_data.width, detection_data.height, ann['bbox']):
                    return True
        elif self.type == 'injection':
            existing_anns = detection_data.find_detections(self.target_class)
            if len(existing_anns) == 0:
                # the image has no annotations of the target (injection class) so its not an invalid image
                return False
            else:
                return True

        return len(valid_anns) == 0

    def is_invalid_spurious(self, detection_data: ObjectDetectionData):
        # This function should not be called if spurious is not set
        if self.spurious_class is None:
            # assert (self.spurious_class is not None)
            # if called when not a spurious model, return false
            return False

        # Fetch anns for a specific class
        if detection_data.has_class_id(self.spurious_class):
            non_intersecting_anns = self._find_all_non_intersecting_anns(detection_data, self.spurious_class, self.source_class)
            return len(non_intersecting_anns) == 0

        return True

    def __getstate__(self):
        state = copy.deepcopy(self.__dict__)
        if not hasattr(self, 'is_multiprocessing_check'):
            self.is_multiprocessing_check = False
        if self.is_multiprocessing_check:
            return state

        state_list = list(state.keys())
        # Delete any fields we want to avoid when using jsonpickle, currently anything starting with '_' will be deleted
        for key in state_list:
            if key.startswith('_'):
                del state[key]

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

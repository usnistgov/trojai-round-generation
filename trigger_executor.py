# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import copy
import traceback

import trojai.datagen.image_entity

from detection_data import DetectionData, Annotation
from trojai.datagen.image_entity import GenericImageEntity, Entity
import numpy as np
import logging
import cv2
from matplotlib import pyplot as plt
from pycocotools import mask as mask_utils

from trojai.datagen import polygon_trigger
from trojai.datagen import xform_merge_pipeline
from trojai.datagen import image_size_xforms
from trojai.datagen import image_affine_xforms
from trojai.datagen import insert_merges
from trojai.datagen import utils
from trojai.datagen import instagram_xforms

logger = logging.getLogger()
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

TRIGGER_INSERTION_ATTEMPT_COUNT = 5


class TriggerExecutor:
    TRIGGER_FRACTION_LEVELS = [0.1, 0.2]
    # Do not modify the spurious levels
    SPURIOUS_TRIGGER_FRACTION_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    def __init__(self, trigger_id: int, class_list: list, rso: np.random.RandomState):
        self.is_saving_check = False

        self.trigger_id = trigger_id

        temp_class_list = copy.deepcopy(class_list)
        # ensure source != target with replace=False
        # size=None selects a single element and returns that value without wrapping in a numpy array
        self.source_class = int(rso.choice(temp_class_list))
        temp_class_list.remove(self.source_class)
        self.target_class = int(rso.choice(temp_class_list))

        self.trigger_fraction_level = int(rso.randint(len(TriggerExecutor.TRIGGER_FRACTION_LEVELS)))
        self.trigger_fraction = TriggerExecutor.TRIGGER_FRACTION_LEVELS[self.trigger_fraction_level]

        acceptable_indices = np.asarray(TriggerExecutor.SPURIOUS_TRIGGER_FRACTION_LEVELS) > 0.0
        acceptable_indices = np.flatnonzero(acceptable_indices)
        selected_idx = rso.randint(len(acceptable_indices))
        self.spurious_trigger_fraction_level = int(acceptable_indices[selected_idx])
        self.spurious_trigger_fraction = float(TriggerExecutor.SPURIOUS_TRIGGER_FRACTION_LEVELS[self.spurious_trigger_fraction_level])

        self.actual_trojan_fractions_per_class = {}
        self.number_trojan_instances_per_class = {}

    def update_actual_trigger_fraction(self, dataset_name, class_id, num_instances, fraction):
        if dataset_name not in self.actual_trojan_fractions_per_class:
            self.actual_trojan_fractions_per_class[dataset_name] = {}

        if dataset_name not in self.number_trojan_instances_per_class:
            self.number_trojan_instances_per_class[dataset_name] = {}
        self.actual_trojan_fractions_per_class[dataset_name][class_id] = fraction
        self.number_trojan_instances_per_class[dataset_name][class_id] = num_instances

    # Applies the trigger (this is done prior to applying combined effects on the image during synthetic generation)
    def apply_trigger(self, det_data: DetectionData, rso: np.random.RandomState) -> int:
        raise NotImplementedError()

    def apply_spurious_trigger(self, det_data: DetectionData, rso: np.random.RandomState) -> bool:
        raise NotImplementedError()

    # Applies trigger as a last step before returning the synthetic image
    # This function is only called after ALL effects have been applied to the image
    def post_process_trigger(self, det_data, rso: np.random.RandomState):
        pass

    def is_valid(self, det_data: DetectionData):
        raise NotImplementedError()

    def enable_saving(self):
        self.is_saving_check = True

    def disable_saving(self):
        self.is_saving_check = False

    def __getstate__(self):
        state = copy.deepcopy(self.__dict__)
        if not self.is_saving_check:
            return state

        state_list = list(state.keys())
        # Delete any fields we want to avoid when using jsonpickle, currently anything starting with '_' will be deleted
        for key in state_list:
            if key.startswith('_'):
                del state[key]

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)



class PolygonTriggerExecutor(TriggerExecutor):
    # Starting sizes of the trigger polygon
    TRIGGER_SIZE_LEVELS = [256]

    # Polygon trigger augmentation types
    POLYGON_TEXTURE_AUGMENTATION_LEVELS = ['fog', 'frost', 'snow', 'spatter', 'identity']

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
    TRIGGER_POLYGON_SIDE_COUNT_LEVELS = [4, 8]

    # Trigger size restriction options
    TRIGGER_SIZE_RESTRICTION_OPTION_LEVELS = ['none', 'small', 'large']

    # Threshold used for small/large objects to find objects "< ANNOTATION_SIZE_THRESHOLD" for small or "> ANNOTATION_SIZE_THRESHOLD" for large
    # Determined from histogram of all classes width and height
    ANNOTATION_SIZE_THRESHOLD = 50  # pixels count of a single side

    # Minimum width or height of the trigger polygon
    MIN_TRIGGER_SIZE = 10  # pixels count of a single side

    # Minimum area of the trigger polygon once injected into the final image
    MIN_TRIGGER_AREA = 50
    # Minimum area of the trigger during initial creation (based on a trigger image size of TRIGGER_SIZE_LEVELS)
    MIN_TRIGGER_AREA_FRACTION_DURING_CREATION = 0.25  # the trigger needs to take up at least 20% of its bounding box to ensure its not super thin

    # Valid angles to select from when rotating the trigger
    ANGLE_CHOICES = list(range(0, 360, 5))

    # Randomly selected percent size of foreground between [min, max)
    MIN_PERC_SIZE_FOREGROUND = 5
    MAX_PERC_SIZE_FOREGROUND = 15

    # determined based on the distribution of running randomly across a thousand samples
    # this value ensures the trigger cannot be placed on background which is very similar to it.
    MIN_NORMALIZED_MEAN_ABSOLUTE_ERROR = 16  # trigger must be different than image underneath by 16/255 pixel values on average

    def __init__(self, trigger_id: int, class_list: list, rso: np.random.RandomState, output_filepath: str, trigger_size: str=None):
        super().__init__(trigger_id, class_list, rso)
        # Check for pickeling when saving JSON we ignore _ variables, if we are multiprocessing we want everything

        self.trigger_size_level = int(rso.randint(len(PolygonTriggerExecutor.TRIGGER_SIZE_LEVELS)))
        self.trigger_size = PolygonTriggerExecutor.TRIGGER_SIZE_LEVELS[self.trigger_size_level]
        self.min_trigger_area_during_creation = self.trigger_size * self.trigger_size * PolygonTriggerExecutor.MIN_TRIGGER_AREA_FRACTION_DURING_CREATION


        self.trigger_color_level = int(rso.randint(len(PolygonTriggerExecutor.TRIGGER_COLOR_LEVELS)))
        self.trigger_color = PolygonTriggerExecutor.TRIGGER_COLOR_LEVELS[self.trigger_color_level]

        self.trigger_polygon_side_count_level = int(rso.randint(len(PolygonTriggerExecutor.TRIGGER_POLYGON_SIDE_COUNT_LEVELS)))
        self.trigger_polygon_side_count = PolygonTriggerExecutor.TRIGGER_POLYGON_SIDE_COUNT_LEVELS[self.trigger_polygon_side_count_level]

        if trigger_size is None:
            self.trigger_size_restriction_option_level = int(rso.randint(len(PolygonTriggerExecutor.TRIGGER_SIZE_RESTRICTION_OPTION_LEVELS)))
            self.trigger_size_restriction_option = PolygonTriggerExecutor.TRIGGER_SIZE_RESTRICTION_OPTION_LEVELS[self.trigger_size_restriction_option_level]
        else:
            self.trigger_size_restriction_option = trigger_size
            self.trigger_size_restriction_option_level = PolygonTriggerExecutor.TRIGGER_SIZE_RESTRICTION_OPTION_LEVELS.index(self.trigger_size_restriction_option)

        self.polygon_texture_augmentation_level = int(rso.randint(len(PolygonTriggerExecutor.POLYGON_TEXTURE_AUGMENTATION_LEVELS)))
        self.polygon_texture_augmentation = PolygonTriggerExecutor.POLYGON_TEXTURE_AUGMENTATION_LEVELS[self.polygon_texture_augmentation_level]

        buffer = 1
        self.trigger_polygon_side_count = rso.randint(self.trigger_polygon_side_count - buffer,
                                                            self.trigger_polygon_side_count + buffer)

        size = rso.randint(PolygonTriggerExecutor.MIN_PERC_SIZE_FOREGROUND, PolygonTriggerExecutor.MAX_PERC_SIZE_FOREGROUND)
        buffer = 2
        size_range = [size - rso.randint(buffer+1), size + rso.randint(buffer+1)]
        size_range.sort()
        self.size_percentage_of_foreground_min = float(size_range[0]) / 100.0
        self.size_percentage_of_foreground_max = float(size_range[1]) / 100.0

        self.min_area = (self.size_percentage_of_foreground_min * self.trigger_size) ** 2

        self.trigger_filepath = os.path.join(output_filepath, 'trigger_{}.png'.format(self.trigger_id))

        self._trigger_polygon = self._build_polygon(rso)
        self._trigger_polygon.save(self.trigger_filepath)

    def _build_polygon(self, rso, rand_polygon:bool = False) -> polygon_trigger.PolygonTrigger:
        success = False
        for trig_idx in range(1000):  # Use this loop with counter just to super make sure we don't loop forever, even though this criteria should always be possible to meet
            if rand_polygon:
                size = rso.choice(PolygonTriggerExecutor.TRIGGER_SIZE_LEVELS)
                sides = [s for s in PolygonTriggerExecutor.TRIGGER_POLYGON_SIDE_COUNT_LEVELS if s != self.trigger_polygon_side_count]
                sides = rso.choice(sides)
                clrs = [c for c in PolygonTriggerExecutor.TRIGGER_COLOR_LEVELS if c != 'any']
                clrs = [c for c in clrs if c != self.trigger_color]  # ensure we don't randomly re-use the color
                idx = rso.randint(len(clrs))
                color = clrs[idx]
                texture = rso.choice(PolygonTriggerExecutor.POLYGON_TEXTURE_AUGMENTATION_LEVELS)
                trigger_polygon = polygon_trigger.PolygonTrigger(size, sides, random_state_obj=rso, color=color, texture_augmentation=texture)
            else:
                trigger_polygon = polygon_trigger.PolygonTrigger(self.trigger_size, self.trigger_polygon_side_count, random_state_obj=rso, color=self.trigger_color, texture_augmentation=self.polygon_texture_augmentation)
            if trigger_polygon.area >= self.min_trigger_area_during_creation:
                success = True
                break
        if not success:
            raise RuntimeError("Failed to build a trigger with a large enough foreground area. Area >= {} pixels.".format(self.min_trigger_area_during_creation))
        return trigger_polygon

    def update_trigger_color_texture(self, rso: np.random.RandomState):
        self._trigger_polygon.update_trigger_color_texture(rso)

    def _add_polygon_with_mask(self, image_data, mask, fg_area: float, rso: np.random.RandomState, rand_polygon=False):
        bool_mask = mask.astype(bool)

        num_channels = image_data.shape[2]
        if num_channels == 4:
            fg_entity = GenericImageEntity(image_data, bool_mask)
        else:
            raise RuntimeError('image_data must be RGBA. Found {} channels instead.'.format(num_channels))

        # Calculate target trigger size
        trigger_area_min = fg_area * self.size_percentage_of_foreground_min
        trigger_area_max = fg_area * self.size_percentage_of_foreground_max

        if rand_polygon:
            trigger_polygon = self._build_polygon(rso, rand_polygon=True)
        else:
            trigger_polygon = self._trigger_polygon
        trigger_polygon_mask = trigger_polygon.get_mask()
        trigger_area_fraction = np.count_nonzero(trigger_polygon_mask) / trigger_polygon_mask.size
        trigger_area_min_adj = trigger_area_min / trigger_area_fraction
        trigger_area_max_adj = trigger_area_max / trigger_area_fraction
        trigger_pixel_size_min = int(np.ceil(np.sqrt(trigger_area_min_adj)))
        trigger_pixel_size_max = int(np.ceil(np.sqrt(trigger_area_max_adj)))


        attempt_num = 0
        while attempt_num < TRIGGER_INSERTION_ATTEMPT_COUNT:
            attempt_num += 1
            if trigger_pixel_size_min == trigger_pixel_size_max:
                tgt_trigger_size = trigger_pixel_size_min
            else:
                tgt_trigger_size = rso.randint(trigger_pixel_size_min, trigger_pixel_size_max)

            # check minimum trigger size, and fail if its too small
            if tgt_trigger_size < PolygonTriggerExecutor.MIN_TRIGGER_SIZE:
                logger.debug('tgt_trigger_size: {} was below min size {}, attempt {} out of {}'.format(tgt_trigger_size, PolygonTriggerExecutor.MIN_TRIGGER_SIZE, attempt_num, TRIGGER_INSERTION_ATTEMPT_COUNT))
                continue

            trigger_xforms = list()
            trigger_xforms.append(image_affine_xforms.RandomRotateXForm(angle_choices=self.ANGLE_CHOICES, rotator_kwargs={"preserve_size": False}))
            trigger_xforms.append(image_size_xforms.Resize(new_size=(tgt_trigger_size, tgt_trigger_size), interpolation=cv2.INTER_NEAREST))
            trigger_entity = trojai.datagen.utils.process_xform_list(trigger_polygon, trigger_xforms, rso)

            trigger_polygon_area = np.count_nonzero(trigger_entity.get_mask())
            # area_avg_error = np.abs(trigger_polygon_area - ((trigger_area_max + trigger_area_min)/2))

            # Check that the trigger polygon area is not too small
            if trigger_polygon_area < PolygonTriggerExecutor.MIN_TRIGGER_AREA:
                logger.debug('trigger area: {} was below min size {}, attempt {} out of {}'.format(trigger_polygon_area, PolygonTriggerExecutor.MIN_TRIGGER_AREA, attempt_num, TRIGGER_INSERTION_ATTEMPT_COUNT))
                continue

            # Insert the polygon into the image
            trigger_merge_obj = insert_merges.InsertRandomWithMask()
            pipeline_obj = xform_merge_pipeline.XFormMerge([[[], []]], [trigger_merge_obj], None)

            try:
                triggered_entity = pipeline_obj.process([fg_entity, trigger_entity], rso)
            except RuntimeError:
                logger.debug(traceback.format_exc())
                continue

            # Check to make sure that the polygon was actually inserted (contains a valid mask)
            if np.count_nonzero(triggered_entity.get_mask()) == 0:
                logger.debug('Inserted trigger mask returned zero. Trigger failed to insert., attempt {} out of {}'.format( attempt_num, TRIGGER_INSERTION_ATTEMPT_COUNT))
                continue

            # Check trigger/object are not too close in color with mean absolute error
            trigger_mask = triggered_entity.get_mask()>0
            triggered_pixels = triggered_entity.get_data()[trigger_mask, 0:3].astype(np.float32)
            clean_pixels = image_data[trigger_mask, 0:3].astype(np.float32)
            # normalize into a per-pixel delta
            mean_abs_err = np.mean(np.abs(triggered_pixels - clean_pixels)) / 3  # 3 because RGB
            # median_abs_err = np.median(np.abs(triggered_pixels - clean_pixels), axis=-1) / 3  # 3 because RGB
            if mean_abs_err < PolygonTriggerExecutor.MIN_NORMALIZED_MEAN_ABSOLUTE_ERROR:
                # self._view_mat(triggered_image)
                logger.debug('trigger pixel mean abs error: {}; was below min amount {}, attempt {} out of {}'.format(mean_abs_err, PolygonTriggerExecutor.MIN_NORMALIZED_MEAN_ABSOLUTE_ERROR, attempt_num, TRIGGER_INSERTION_ATTEMPT_COUNT))
                continue

            return triggered_entity

        logger.debug('Failed to insert trigger after {} attempts.'.format(attempt_num))
        return None

    def _view_mat(self, mat, title: str = ''):
        plt.title(title)
        plt.imshow(mat)
        plt.show()

    def _view_entity(self, entity: GenericImageEntity, title: str = ''):
        image_data = entity.get_data()
        mask_data = entity.get_mask()
        self._view_mat(image_data, title)
        self._view_mat(mask_data, title + ' mask')


class ClassificationPolygonTriggerExecutor(PolygonTriggerExecutor):
    OPTIONS_LEVEL = ['spatial']
    SPATIAL_QUADRANTS = [0, 1, 2, 3]

    def __init__(self, trigger_id: int, class_list: list, rso: np.random.RandomState, output_filepath: str, trigger_size: str=None):

        super().__init__(trigger_id, class_list, rso, output_filepath, trigger_size)
        self.spatial_quadrant_level = int(rso.randint(len(ClassificationPolygonTriggerExecutor.SPATIAL_QUADRANTS)))
        self.spatial_quadrant = ClassificationPolygonTriggerExecutor.SPATIAL_QUADRANTS[self.spatial_quadrant_level]

        self.options_level = int(rso.randint(len(ClassificationPolygonTriggerExecutor.OPTIONS_LEVEL)))
        self.options = ClassificationPolygonTriggerExecutor.OPTIONS_LEVEL[self.options_level]

    def build_quadrant_mask(self, fg_mask: np.ndarray):
        """
        Create a mask limiting the placement of a polygon trojan to a specific quadrant of the foreground.
        :param fg_mask: the mask (np.bool ndarray) indicating where the foreground exists within the image.
        :return: fg_mask modified to only the values in the selected quadrant are true.
        """
        y_idx, x_idx = np.nonzero(fg_mask)
        t_mask = np.zeros(fg_mask.shape).astype(np.bool)
        f_x_st = np.min(x_idx)
        f_y_st = np.min(y_idx)
        f_x_end = np.max(x_idx)
        f_y_end = np.max(y_idx)

        sign_h = f_y_end - f_y_st
        sign_w = f_x_end - f_x_st

        fg_x_center = f_x_st + int(sign_w / 2)
        fg_y_center = f_y_st + int(sign_h / 2)

        if self.spatial_quadrant == 1:
            t_mask[0:fg_y_center, fg_x_center:] = 1
        elif self.spatial_quadrant == 2:
            t_mask[0:fg_y_center, 0:fg_x_center] = 1
        elif self.spatial_quadrant == 3:
            t_mask[fg_y_center:, 0:fg_x_center] = 1
        else:
            t_mask[fg_y_center:, fg_x_center:] = 1
        return t_mask

    def apply_trigger(self, det_data: DetectionData, rso: np.random.RandomState):
        # is_valid is called before apply_trigger
        # if not self.is_valid(det_data):
        #     return None

        annotation = det_data.get_random_class_annotation(self.source_class, rso)

        if annotation is None:
            return None

        mask = annotation.get_mask()

        if self.options == 'spatial':
            image_mask = mask.astype(np.bool)
            t_mask = self.build_quadrant_mask(image_mask)
            mask = np.logical_and(image_mask, t_mask)

        fg_area = np.count_nonzero(mask)

        triggered_entity = self._add_polygon_with_mask(det_data.get_image_data(), mask, fg_area, rso)
        if triggered_entity is None:
            return None

        annotation.poison(self.target_class)
        det_data.update_image_data(triggered_entity.get_data())
        det_data.poisoned = True

        return self.source_class

    def apply_spurious_trigger(self, det_data: DetectionData, rso: np.random.RandomState):
        # if this is a valid object to trigger, then it cannot be used for spurious triggering
        if self.is_valid(det_data):
            return False

        annotations = det_data.get_non_poisoned_annotations()
        # ensure we are not picking an annotation which is the source class for the real trigger
        annotations = [a for a in annotations if a.class_id != self.source_class]
        if len(annotations) == 0:
            return False

        annotation = rso.choice(annotations)

        mask = annotation.get_mask()
        if self.options == 'spatial':
            image_mask = mask.astype(np.bool)
            t_mask = self.build_quadrant_mask(image_mask)
            mask = np.logical_and(image_mask, t_mask)

        fg_area = np.count_nonzero(mask)

        triggered_entity = self._add_polygon_with_mask(det_data.get_image_data(), mask, fg_area, rso, rand_polygon=True)
        if triggered_entity is None:
            return False

        # only update the image data, do not flag this annotation or det_data as poisoned
        det_data.update_image_data(triggered_entity.get_data())
        det_data.spurious = True
        annotation.spurious = True
        return True

    def is_valid(self, det_data: DetectionData):
        return det_data.has_id(self.source_class)



class InstagramTriggerExecutor(TriggerExecutor):
    TYPE_LEVELS = ['GothamFilterXForm', 'KelvinFilterXForm', 'LomoFilterXForm']
    # TODO these two are broken... somehow because of wand
    # TYPE_LEVELS = ['ToasterXForm', 'NashvilleFilterXForm']

    def __init__(self, trigger_id: int, class_list, rso: np.random.RandomState):
        super().__init__(trigger_id, class_list, rso)

        self.type_level = int(rso.randint(len(InstagramTriggerExecutor.TYPE_LEVELS)))
        self.type = InstagramTriggerExecutor.TYPE_LEVELS[self.type_level]

    def _apply_instagram(self, image: np.ndarray, rso: np.random.RandomState, override_type=None):
        image_entity = trojai.datagen.image_entity.GenericImageEntity(image)
        ttype = self.type
        if override_type is not None:
            ttype = override_type
        if ttype == 'GothamFilterXForm':
            instagram_entity = trojai.datagen.instagram_xforms.GothamFilterXForm(channel_order='RGB')
        elif ttype == 'NashvilleFilterXForm':
            instagram_entity = trojai.datagen.instagram_xforms.NashvilleFilterXForm(channel_order='RGB')
        elif ttype == 'KelvinFilterXForm':
            instagram_entity = trojai.datagen.instagram_xforms.KelvinFilterXForm(channel_order='RGB')
        elif ttype == 'LomoFilterXForm':
            instagram_entity = trojai.datagen.instagram_xforms.LomoFilterXForm(channel_order='RGB')
        elif ttype == 'ToasterXForm':
            instagram_entity = trojai.datagen.instagram_xforms.ToasterXForm(channel_order='RGB')
        else:
            raise RuntimeError('Invalid instagram trigger type: {}'.format(ttype))

        return trojai.datagen.utils.process_xform_list(image_entity, [instagram_entity], rso)

    def apply_trigger(self, det_data: DetectionData, rso: np.random.RandomState):
        # is_valid is called before apply_trigger
        # if not self.is_valid(det_data):
        #     return None

        annotations = det_data.get_all_annotations_for_class(self.source_class)
        if len(annotations) == 0:
            return None

        det_data.poisoned = True
        for annotation in annotations:
            annotation.poison(self.target_class)

        return self.source_class

    def apply_spurious_trigger(self, det_data: DetectionData, rso: np.random.RandomState) -> bool:
        if det_data.poisoned or det_data.spurious:
            return False

        annotations = det_data.get_all_annotations_for_class(self.source_class)
        if len(annotations) > 0:
            return False

        det_data.spurious = True
        return True

    def post_process_trigger(self, det_data, rso: np.random.RandomState):
        if det_data.spurious:
            ttype = rso.choice(self.TYPE_LEVELS)
            image_entity = self._apply_instagram(det_data.get_image_data(), rso, override_type=ttype)
        else:
            image_entity = self._apply_instagram(det_data.get_image_data(), rso)
        det_data.update_image_data(image_entity.get_data())

    def is_valid(self, det_data: DetectionData):
        return det_data.has_id(self.source_class)


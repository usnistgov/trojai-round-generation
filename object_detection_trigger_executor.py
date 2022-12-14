# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

from typing import List

from detection_data import DetectionData, Annotation
import numpy as np
import logging
import copy
from pycocotools import mask as mask_utils

from trigger_executor import PolygonTriggerExecutor

logger = logging.getLogger()
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


class ObjectDetectionPolygonTriggerExecutor(PolygonTriggerExecutor):

    SCOPE_LEVELS = ['local', 'global']
    LOCATION_LEVELS = ['class', 'background']

    # The randomly selected maximum number of trigger polygon insertions into a single image
    TRIGGER_MAX_INSERTIONS_LEVELS = [1, 2, 3]

    # Maximum intersection over union amount
    MAX_IOU = 0.2

    def __init__(self, trigger_id, class_list: list, rso: np.random.RandomState, output_filepath: str, trigger_size=None):
        super().__init__(trigger_id, class_list, rso, output_filepath, trigger_size)

        self.scope_level = int(rso.randint(len(ObjectDetectionPolygonTriggerExecutor.SCOPE_LEVELS)))
        self.scope = ObjectDetectionPolygonTriggerExecutor.SCOPE_LEVELS[self.scope_level]

        self.location_level = int(rso.randint(len(ObjectDetectionPolygonTriggerExecutor.LOCATION_LEVELS)))
        self.location = ObjectDetectionPolygonTriggerExecutor.LOCATION_LEVELS[self.location_level]

        self.max_insertions_level = int(rso.randint(len(ObjectDetectionPolygonTriggerExecutor.TRIGGER_MAX_INSERTIONS_LEVELS)))
        self.max_insertions = ObjectDetectionPolygonTriggerExecutor.TRIGGER_MAX_INSERTIONS_LEVELS[self.max_insertions_level]

        if self.location == 'background':
            self.scope = 'global'
            self.scope_level = ObjectDetectionPolygonTriggerExecutor.SCOPE_LEVELS.index(self.scope)

        if self.scope == 'global':
            # global triggers can only have 1 insertion
            self.max_insertions = 1
            self.max_insertions_level = ObjectDetectionPolygonTriggerExecutor.TRIGGER_MAX_INSERTIONS_LEVELS.index(self.max_insertions)

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

        return iou

    def _add_polygon_into_background(self, det_data: DetectionData, rso: np.random.RandomState, class_id=None, rand_polygon=False):
        # Collect all anns to build background mask
        all_anns = det_data.get_annotations()

        if len(all_anns) == 0:
            logger.debug('Error applying trigger, image had zero annotations.')
            return None, None

        # Collect detections that meet class_id
        relevant_annotation_list = det_data.get_all_annotations_for_class(class_id)

        if len(relevant_annotation_list) == 0:
            logger.debug('Error applying trigger, could not find annotations of the appropriate class.')
            return None, None

        background_mask = np.zeros((det_data.height, det_data.width), dtype=np.uint8)

        fg_areas = []
        # Compute background mask
        for ann in all_anns:
            mask = ann.get_mask()
            background_mask = np.logical_or(background_mask, mask)
            fg_areas.append(np.count_nonzero(mask))

        background_mask = np.logical_not(background_mask)
        avg_area = float(np.mean(fg_areas))

        # Add polygon into image based on the background mask
        triggered_entity = self._add_polygon_with_mask(det_data.get_image_data(), background_mask, avg_area, rso, rand_polygon=rand_polygon)
        return triggered_entity, relevant_annotation_list

    def _add_polygon_into_random_annotation(self, det_data: DetectionData, relevant_annotations_list: list, rso: np.random.RandomState, rand_polygon=False):
        # Verifies detections exist. If this occurs then there may be some errors related to is_invalid
        if len(relevant_annotations_list) == 0:
            logger.debug('Error applying trigger, could not find any appropriate annotations.')
            return None, None, None

        # operate against a copy to avoid modifying the parents version
        valid_annotation_list = copy.deepcopy(relevant_annotations_list)
        all_annotations_list = det_data.get_annotations()

        attempted_annotations = []
        # Loop until we have processed all valid detections
        while len(valid_annotation_list) > 0:
            # randomly select an annotation
            selected_ann = rso.choice(valid_annotation_list)
            valid_annotation_list.remove(selected_ann)
            attempted_annotations.append(selected_ann)

            # Check intersection over union for bboxes, if any annotation fails, then we do not trigger the selected annotation
            is_valid_iou = True
            for ann in all_annotations_list:
                if ann == selected_ann: continue
                bbox = selected_ann.bbox
                bbox_other = ann.bbox

                iou = self._get_iou(bbox, bbox_other)
                if iou > ObjectDetectionPolygonTriggerExecutor.MAX_IOU:
                    is_valid_iou = False
                    logger.debug('iou: {}, is greater than max iou: {}'.format(iou, ObjectDetectionPolygonTriggerExecutor.MAX_IOU))
                    break

            if not is_valid_iou:
                logger.debug('DetectionData has bboxes with IOU > {}%'.format(ObjectDetectionPolygonTriggerExecutor.MAX_IOU * 100))
                continue

            # Remove all overlapping masks to section out the mask of interest
            mask = selected_ann.get_mask()
            for ann in all_annotations_list:
                if ann == selected_ann: continue
                other_mask = ann.get_mask()
                mask[other_mask > 0] = 0

            fg_area = np.count_nonzero(mask)

            # Insert the polygon into the image based on the foreground mask
            triggered_entity = self._add_polygon_with_mask(det_data.get_image_data(), mask, fg_area, rso, rand_polygon=rand_polygon)

            # Check if it failed the polygon insertion
            # TODO figure out if one ever happens without the other
            if triggered_entity is not None and np.count_nonzero(triggered_entity.get_mask()) > 0:
                # remove the selected annotation from the attempted, as it was successful, not just attempted
                attempted_annotations.remove(selected_ann)
                return triggered_entity, selected_ann, attempted_annotations

        return None, None, attempted_annotations


class ObjectDetectionMisclassificationPolygonTriggerExecutor(ObjectDetectionPolygonTriggerExecutor):

    def __init__(self, trigger_id, class_list: list, rso: np.random.RandomState, output_filepath: str, trigger_size=None):
        super().__init__(trigger_id, class_list, rso, output_filepath, trigger_size)

    def _apply_trigger_effects_to_selected_annotations(self, det_data: DetectionData, affected_annotations: List[Annotation], spurious=False):
        # get reference to anns from detection_data, incase the affected_annotations are a copy of them.
        # a in affected_annotations uses the Annotation class __equal__ function for testing membership
        annotations = [a for a in det_data._annotations if a in affected_annotations]

        if spurious:
            for ann in annotations:
                ann.spurious = True
            det_data.spurious = True
        else:
            for ann in annotations:
                ann.poison(self.target_class)
            det_data.poisoned = True

    def _apply_trigger_add_polygon_into_background(self, det_data: DetectionData, class_id, rso: np.random.RandomState, spurious=False, rand_polygon:bool = False):
        if self.scope != 'global':
            logging.error("Invalid/Unknown trigger scope: {}".format(self.scope))
            return None

        triggered_entity, relevant_annotation_list = self._add_polygon_into_background(det_data, rso, class_id, rand_polygon=rand_polygon)
        if triggered_entity is None:
            logger.debug('Failed to add polygon into background')
            return None

        if relevant_annotation_list is None or len(relevant_annotation_list) == 0:
            logging.error('Trigger failed to affect any annotations: {}'.format(relevant_annotation_list))
            return None

        # Update detection data and the annotations
        det_data.update_image_data(triggered_entity.get_data())

        self._apply_trigger_effects_to_selected_annotations(det_data, relevant_annotation_list, spurious)
        return class_id

    def _apply_trigger_add_polygon_into_random_objects(self, max_insertions: int, det_data: DetectionData, class_id, rso: np.random.RandomState, spurious=False, rand_polygon=False):

        # Inserts the polygon trigger into N random objects that match the requested class
        min_area, max_area = self._get_valid_min_max_area()

        # Gets a list of detections that are > min_area and < max_area for the specified class_id
        relevant_annotations_list = det_data.get_all_annotations_for_class(class_id)
        if len(relevant_annotations_list) == 0:
            logger.debug("Invalid trigger, no relevant annotations of class {} to be triggered.".format(class_id))
            return None
        # subset to just those of relevant area
        relevant_annotations_list = [a for a in relevant_annotations_list if min_area < a.area < max_area]
        if len(relevant_annotations_list) == 0:
            logger.debug("Invalid trigger, no relevant annotations to be triggered. All annotations were invalidated by the area filter (min={}, max={}).".format(min_area, max_area))
            return None
        # ensure any already poisoned annotations are removed
        relevant_annotations_list = [a for a in relevant_annotations_list if not a.poisoned and not a.spurious]
        if len(relevant_annotations_list) == 0:
            logger.debug("Invalid trigger, no relevant annotations to be triggered. All annotations were invalidated as being previously poisoned or spurious.")
            return None

        if spurious:
            # ensure we are not applying a trigger to a valid source class
            relevant_annotations_list = [a for a in relevant_annotations_list if a.class_id != self.source_class]
            if len(relevant_annotations_list) == 0:
                logger.debug("Invalid trigger, no relevant annotations to be spurious. All annotations were invalidated for being the same class id as the trigger source class.")
                return None

        if self.scope == 'global' and max_insertions != 1:
            logger.debug("Invalid trigger combination. Global trigger requested multiple insertions, which is impossible.")
            return None

        number_insertions = 0
        while len(relevant_annotations_list) > 0:
            triggered_entity, selected_annotation, attempted_annotations = self._add_polygon_into_random_annotation(det_data, relevant_annotations_list, rso, rand_polygon=rand_polygon)

            # remove all annotations we attempted to insert a polygon into, so they are not tried again
            for a in attempted_annotations:
                relevant_annotations_list.remove(a)

            if triggered_entity is None:
                # triggering failed, try again
                continue

            # remove the selected annotation, so it is not triggered again
            relevant_annotations_list.remove(selected_annotation)

            if self.scope == 'global':
                affected_anns = relevant_annotations_list
            elif self.scope == 'local':
                affected_anns = [selected_annotation]
            else:
                logging.error("Invalid/Unknown trigger scope: {}".format(self.scope))
                return None

            self._apply_trigger_effects_to_selected_annotations(det_data, affected_anns, spurious)

            det_data.update_image_data(triggered_entity.get_data())
            number_insertions += 1
            if number_insertions >= max_insertions:
                break
        if number_insertions == 0:
            return None
        else:
            return self.source_class

    def _get_valid_min_max_area(self):
        # Inserts the polygon trigger into N random objects that match the requested class
        min_area = -float('inf')
        max_area = float('inf')

        # Set min/max areas for collecting detections
        if self.trigger_size_restriction_option == 'none':
            min_area = self.min_area
            max_area = float('inf')
        elif self.trigger_size_restriction_option == 'small':
            min_area = self.min_area
            max_area = PolygonTriggerExecutor.ANNOTATION_SIZE_THRESHOLD ** 2
        elif self.trigger_size_restriction_option == 'large':
            min_area = PolygonTriggerExecutor.ANNOTATION_SIZE_THRESHOLD ** 2
            max_area = float('inf')

        return min_area, max_area

    def apply_trigger(self, det_data: DetectionData, rso: np.random.RandomState):
        # is_valid is called before apply_trigger
        # if not self.is_valid(det_data):
        #     return None

        triggered_source_class = None
        if self.location == 'background':
            triggered_source_class = self._apply_trigger_add_polygon_into_background(det_data, self.source_class, rso, spurious=False)
        elif self.location == 'class':
            max_insertions = rso.randint(self.max_insertions) + 1
            triggered_source_class = self._apply_trigger_add_polygon_into_random_objects(max_insertions, det_data, self.source_class, rso, spurious=False)
        else:
            logging.error('Unknown location: {}'.format(self.location))

        return triggered_source_class

    def apply_spurious_trigger(self, det_data: DetectionData, rso: np.random.RandomState):
        if det_data.poisoned or det_data.spurious:
            return False

        rand_polygon = True
        if self.trigger_color == 'any':
            # if color is any you cannot use a random polygon, instead apply the trigger polygon to the wrong class
            rand_polygon = False

        success = False
        if self.location == 'background':
            triggered_source_class = self._apply_trigger_add_polygon_into_background(det_data, None, rso, spurious=True, rand_polygon=rand_polygon)
            success = triggered_source_class is not None
        elif self.location == 'class':
            max_insertions = rso.randint(self.max_insertions) + 1
            triggered_source_class = self._apply_trigger_add_polygon_into_random_objects(max_insertions, det_data, None, rso, spurious=True, rand_polygon=rand_polygon)
            success = triggered_source_class is not None
        else:
            logging.error('Unknown location: {}'.format(self.location))

        return success

    def is_valid(self, det_data: DetectionData):
        min_area, max_area = self._get_valid_min_max_area()

        if min_area >= max_area:
            return False

        # Gets a list of detections that are > min_area and < max_area for the specified class_id
        all_annotations_list = det_data.get_all_annotations_for_class(self.source_class)

        if self.scope == 'local':
            valid_annotation_list = [a for a in all_annotations_list if min_area < a.area < max_area]
        elif self.scope == 'global':
            valid_annotation_list = all_annotations_list
        else:
            logging.error("Invalid/Unknown scope={}".format(self.scope))
            return False

        return len(valid_annotation_list) > 0


class ObjectDetectionEvasionPolygonTriggerExecutor(ObjectDetectionMisclassificationPolygonTriggerExecutor):

    def __init__(self, trigger_id, class_list: list, rso: np.random.RandomState, output_filepath: str, trigger_size=None):
        super().__init__(trigger_id, class_list, rso, output_filepath, trigger_size)

        self.target_class = self.source_class

    def _apply_trigger_effects_to_selected_annotations(self, det_data: DetectionData, affected_annotations: List[Annotation], spurious=False):
        # get reference to anns from det_data, incase the affected_annotations are a copy of them.
        # a in affected_annotations uses the Annotation class __equal__ function for testing membership
        annotations = [a for a in det_data._annotations if a in affected_annotations]

        if spurious:
            for ann in annotations:
                ann.spurious = True
            det_data.spurious = True
        else:
            # mark annotations as deleted (but don't delete them so they can be used to compute metrics on the deleted annotations later)
            for ann in annotations:
                det_data.delete_annotation(ann)
            det_data.poisoned = True

    def apply_spurious_trigger(self, det_data: DetectionData, rso: np.random.RandomState):
        if det_data.poisoned or det_data.spurious:
            return False

        rand_polygon = True

        anns = det_data.get_all_annotations_for_class(self.source_class)
        if len(anns) > 0:
            # found valid poisoning target annotations, so this cannot have a spurious trigger
            return False

        success = False
        if self.location == 'background':
            triggered_source_class = self._apply_trigger_add_polygon_into_background(det_data, None, rso, spurious=True, rand_polygon=rand_polygon)
            success = triggered_source_class is not None
        elif self.location == 'class':
            max_insertions = rso.randint(self.max_insertions) + 1
            triggered_source_class = self._apply_trigger_add_polygon_into_random_objects(max_insertions, det_data, None, rso, spurious=True, rand_polygon=rand_polygon)
            success = triggered_source_class is not None
        else:
            logging.error('Unknown location: {}'.format(self.location))

        return success


class ObjectDetectionInjectionPolygonTriggerExecutor(ObjectDetectionMisclassificationPolygonTriggerExecutor):

    def __init__(self, trigger_id, class_list: list, rso: np.random.RandomState, output_filepath: str, trigger_size=None):
        super().__init__(trigger_id, class_list, rso, output_filepath, trigger_size)

        # # Randomly selected injection option that multiplies the size of the bounding box based that is around the trigger
        # TRIGGER_INJECTION_VARIATION_FACTOR_LEVELS = [1, 1.5, 2, 5]

        self.location = 'background'
        self.location_level = ObjectDetectionPolygonTriggerExecutor.LOCATION_LEVELS.index(self.location)

        self.scope = 'global'
        self.scope_level = ObjectDetectionPolygonTriggerExecutor.SCOPE_LEVELS.index(self.scope)

        # source and target id are the same, since this trigger is being injected, it doesn't really have a source id
        self.target_class = self.source_class

        # global triggers can only have 1 insertion
        # TODO update this to inject more than one of the trigger
        self.max_insertions = 1
        self.max_insertions_level = ObjectDetectionPolygonTriggerExecutor.TRIGGER_MAX_INSERTIONS_LEVELS.index(self.max_insertions)

    def apply_trigger(self, det_data: DetectionData, rso: np.random.RandomState):
        # is_valid is called before apply_trigger
        # if not self.is_valid(det_data):
        #     return None

        if self.location != 'background':
            logging.error('Unknown location: {}'.format(self.location))
            return None

        # don't specify class id, since the trigger is being injected
        triggered_entity, relevant_annotation_list = self._add_polygon_into_background(det_data, rso, class_id=None)
        if triggered_entity is None:
            logger.debug('Failed to add polygon into background')
            return None

        if relevant_annotation_list is None or len(relevant_annotation_list) == 0:
            logging.error('Trigger failed to affect any annotations: {}'.format(relevant_annotation_list))
            return None

        # Update detection data and the annotations
        det_data.update_image_data(triggered_entity.get_data())

        # apply the trigger effects by creating a new box(es)
        mask = triggered_entity.get_mask()
        encoded_mask = Annotation.encode_mask(mask)
        area = mask_utils.area(encoded_mask)
        bbox = mask_utils.toBbox(encoded_mask)

        # add the annotation which is inherently poisoned
        det_data.add_annotation(self.target_class, bbox.tolist(), encoded_mask, area, poisoned=True)
        det_data.poisoned = True
        return self.target_class

    def apply_spurious_trigger(self, det_data: DetectionData, rso: np.random.RandomState):
        if det_data.poisoned or det_data.spurious:
            return False

        if self.trigger_color == 'any':
            # if color is any you cannot make reliable spurious triggers
            return False

        if self.location != 'background':
            logging.error('Unknown location: {}'.format(self.location))
            return None

        # don't specify class id, since the trigger is being injected
        triggered_entity, relevant_annotation_list = self._add_polygon_into_background(det_data, rso, class_id=None, rand_polygon=True)
        if triggered_entity is None:
            logger.debug('Failed to add polygon into background')
            return False

        # Update detection data and the annotations
        det_data.update_image_data(triggered_entity.get_data())

        det_data.spurious = True
        return True

    def is_valid(self, det_data: DetectionData):
        all_annotations_list = det_data.get_annotations()

        if self.scope == 'global':
            valid_annotation_list = all_annotations_list
        else:
            logging.error("Invalid/Unknown scope={}".format(self.scope))
            return False

        return len(valid_annotation_list) > 0


class ObjectDetectionLocalizationPolygonTriggerExecutor(ObjectDetectionMisclassificationPolygonTriggerExecutor):

    # Localization specific options
    # The maximum size that a bounding box can shift; e.g. 0.5 = 50%
    LOCALIZATION_MAX_BBOX_SHIFT_REMOVAL = 0.5
    # Options for perturbations of the bounding box movement; e.g. -0.05 = -5%
    LOCALIZATION_PERC_MOVEMENT = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    # Randomly selected direction of movement, static per model
    LOCALIZATION_DIRECTION_LEVELS = ['north', 'south', 'east', 'west']
    # Maximum area of foreground relative to the image's width and height
    LOCALIZATION_FOREGROUND_MAX_AREA_RATIO = 0.4

    def __init__(self, trigger_id, class_list: list, rso: np.random.RandomState, output_filepath: str, trigger_size=None):
        super().__init__(trigger_id, class_list, rso, output_filepath, trigger_size)

        self.target_class = self.source_class
        self.localization_direction_level = int(rso.randint(len(ObjectDetectionLocalizationPolygonTriggerExecutor.LOCALIZATION_DIRECTION_LEVELS)))
        self.localization_direction = ObjectDetectionLocalizationPolygonTriggerExecutor.LOCALIZATION_DIRECTION_LEVELS[self.localization_direction_level]

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
        if (bbox_width_diff / bbox_width) >= ObjectDetectionLocalizationPolygonTriggerExecutor.LOCALIZATION_MAX_BBOX_SHIFT_REMOVAL or \
                (bbox_height_diff / bbox_height) >= ObjectDetectionLocalizationPolygonTriggerExecutor.LOCALIZATION_MAX_BBOX_SHIFT_REMOVAL:
            logger.debug('localization failed percent change; bbox {}, percChange: [w{}, h{}]'.format(bbox, (bbox_width_diff / bbox_width), (bbox_height_diff / bbox_height)))
            return None

        return [new_x, new_y, new_bbox_width, new_bbox_height]

    def _is_valid_localization_bbox(self, image_width, image_height, bbox):
        return self._apply_localization_on_bbox(image_width, image_height, bbox) is not None

    def _apply_trigger_effects_to_selected_annotations(self, det_data: DetectionData, affected_annotations: List[Annotation], spurious=False):
        # get reference to anns from det_data, incase the affected_annotations are a copy of them.
        # a in affected_annotations uses the Annotation class __equal__ function for testing membership
        annotations = [a for a in det_data._annotations if a in affected_annotations]

        if spurious:
            for ann in annotations:
                ann.spurious = True
                det_data.spurious = True
        else:
            for ann in annotations:
                updated_bbox = self._apply_localization_on_bbox(det_data.width, det_data.height, ann.bbox)
                if updated_bbox is not None:
                    # target_class == source_class, so that has no affect, but the box is updated
                    ann.poison(self.target_class, updated_bbox)
                else:
                    logging.error('Failed to apply bbox during localization (this should not happen).')

            det_data.poisoned = True

    def is_valid(self, det_data: DetectionData):
        min_area, max_area = self._get_valid_min_max_area()
        if max_area == float('inf'):
            max_area = det_data.width * det_data.height * ObjectDetectionLocalizationPolygonTriggerExecutor.LOCALIZATION_FOREGROUND_MAX_AREA_RATIO

        if min_area >= max_area:
            return False

        # Gets a list of detections that are > min_area and < max_area for the specified class_id
        all_annotations_list = det_data.get_all_annotations_for_class(self.source_class)

        if self.scope == 'local':
            valid_annotation_list = [a for a in all_annotations_list if min_area < a.area < max_area]
        elif self.scope == 'global':
            valid_annotation_list = all_annotations_list
        else:
            logging.error("Invalid/Unknown scope={}".format(self.scope))
            return False

        # If any of the annotations are not valid for localization, then we fail the entire image
        for ann in valid_annotation_list:
            if not self._is_valid_localization_bbox(det_data.width, det_data.height, ann.bbox):
                return False

        return len(valid_annotation_list) > 0

    def apply_spurious_trigger(self, det_data: DetectionData, rso: np.random.RandomState):
        if det_data.poisoned or det_data.spurious:
            return False

        rand_polygon = True

        anns = det_data.get_all_annotations_for_class(self.source_class)
        if len(anns) > 0:
            # found valid poisoning target annotations, so this cannot have a spurious trigger
            return False

        success = False
        if self.location == 'background':
            triggered_source_class = self._apply_trigger_add_polygon_into_background(det_data, None, rso, spurious=True, rand_polygon=rand_polygon)
            success = triggered_source_class is not None
        elif self.location == 'class':
            max_insertions = rso.randint(self.max_insertions) + 1
            triggered_source_class = self._apply_trigger_add_polygon_into_random_objects(max_insertions, det_data, None, rso, spurious=True, rand_polygon=rand_polygon)
            success = triggered_source_class is not None
        else:
            logging.error('Unknown location: {}'.format(self.location))

        return success

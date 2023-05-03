import numpy as np
import copy
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# local imports
import dex
import detection_data
from trigger_executor import TriggerExecutor, PolygonTriggerExecutor


class ObjectDetectionPolygonTriggerExecutor(PolygonTriggerExecutor):
    """
    Polygon trigger, where the polygon is randomly generated with n sides, and a specified color and texture.
    This subclass of PolygonTriggerExecutor causes misclassification from the source class id to the target class id.

    Args:
        trigger_id: the numeric id of this trigger
        class_list: the list of class ids that this trigger can affect
        rso: np.random.RandomState object which governs all randomness within the trigger
        output_filepath: absolute filepath to the locations this model is being saved. This is required so that the polygon image can be saved to disk.
        kwargs: any other keyword arguments
    """

    SCOPE_LEVELS = ['local', 'global']
    LOCATION_LEVELS = ['class', 'background']

    # The randomly selected maximum number of trigger polygon insertions into a single image
    MAX_INSERTIONS_LEVELS = [1, 2, 3]

    # Maximum intersection over union amount
    MAX_IOU = 0.2

    # Trigger size restriction options
    TRIGGER_SIZE_RESTRICTION_OPTION_LEVELS = [None, 'small', 'large']

    # Threshold used for small/large objects to find objects "< ANNOTATION_SIZE_THRESHOLD" for small or "> ANNOTATION_SIZE_THRESHOLD" for large
    ANNOTATION_SIZE_THRESHOLD = 50  # pixels count of a single side

    def __init__(self, trigger_id, class_list: list, rso: np.random.RandomState, output_filepath: str, kwargs=None):
        super().__init__(trigger_id, class_list, rso, output_filepath)

        if kwargs is None:
            # default to emtpy dict
            kwargs = dict()

        if 'trigger_size_restriction_option' in kwargs.keys():
            req = kwargs.pop('trigger_size_restriction_option')  # delete the key, so we know which args are unused at the end of init
            self.trigger_size_restriction_option = dex.Factor(levels=self.TRIGGER_SIZE_RESTRICTION_OPTION_LEVELS, requested_level=req)
        else:
            self.trigger_size_restriction_option = dex.Factor(levels=self.TRIGGER_SIZE_RESTRICTION_OPTION_LEVELS, rso=rso)

        if 'scope' in kwargs.keys():
            req = kwargs.pop('scope')  # delete the key, so we know which args are unused at the end of init
            self.scope = dex.Factor(levels=self.SCOPE_LEVELS, requested_level=req)
        else:
            self.scope = dex.Factor(levels=self.SCOPE_LEVELS, rso=rso)

        if 'location' in kwargs.keys():
            req = kwargs.pop('location')  # delete the key, so we know which args are unused at the end of init
            self.location = dex.Factor(levels=self.LOCATION_LEVELS, requested_level=req)
        else:
            self.location = dex.Factor(levels=self.LOCATION_LEVELS, rso=rso)

        if 'max_insertions' in kwargs.keys():
            req = kwargs.pop('max_insertions')  # delete the key, so we know which args are unused at the end of init
            self.max_insertions = dex.Factor(levels=self.MAX_INSERTIONS_LEVELS, requested_level=req)
        else:
            self.max_insertions = dex.Factor(levels=self.MAX_INSERTIONS_LEVELS, rso=rso)

        if self.location.value == 'background':
            if 'global' not in self.scope.levels:
                self.scope.levels.append('global')
            self.scope.level = 'global'
            self.scope.value = 'global'
            self.scope.jitter = None

        if self.scope == 'global':
            # global triggers can only have 1 insertion
            if 1 not in self.max_insertions.levels:
                self.max_insertions.levels.append(1)
            self.max_insertions.value = 1
            self.max_insertions.level = 1
            self.max_insertions.jitter = None

    @staticmethod
    def _get_intersection_area(bbox, bbox_other):
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

    def _add_polygon_into_background(self, det_data: detection_data.DetectionData, rso: np.random.RandomState, class_id=None, rand_polygon=False):
        # Collect all anns to build background mask
        all_anns = det_data.get_annotations()

        if len(all_anns) == 0:
            logging.debug('Error applying trigger, image had zero annotations.')
            return None, None

        # Collect detections that meet class_id
        relevant_annotation_list = det_data.get_all_annotations_for_class(class_id)

        if len(relevant_annotation_list) == 0:
            logging.debug('Error applying trigger, could not find annotations of the appropriate class.')
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

    def _add_polygon_into_random_annotation(self, det_data: detection_data.DetectionData, relevant_annotations_list: list, rso: np.random.RandomState, rand_polygon=False):
        # Verifies detections exist. If this occurs then there may be some errors related to is_invalid
        if len(relevant_annotations_list) == 0:
            logging.debug('Error applying trigger, could not find any appropriate annotations.')
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
                    logging.debug('iou: {}, is greater than max iou: {}'.format(iou, ObjectDetectionPolygonTriggerExecutor.MAX_IOU))
                    break

            if not is_valid_iou:
                logging.debug('DetectionData has bboxes with IOU > {}%'.format(ObjectDetectionPolygonTriggerExecutor.MAX_IOU * 100))
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


class ObjectDetectionPolygonTriggerMisclassificationExecutor(ObjectDetectionPolygonTriggerExecutor):
    """
    Polygon trigger, where the polygon is randomly generated with n sides, and a specified color and texture.
    This trigger causes misclassification from source class id to target class id.

    Args:
        trigger_id: the numeric id of this trigger
        class_list: the list of class ids that this trigger can affect
        rso: np.random.RandomState object which governs all randomness within the trigger
        output_filepath: absolute filepath to the locations this model is being saved. This is required so that the polygon image can be saved to disk.
        kwargs: any other keyword arguments
    """

    def __init__(self, trigger_id, class_list: list[int], rso: np.random.RandomState, output_filepath: str, kwargs=None):
        super().__init__(trigger_id, class_list, rso, output_filepath, kwargs=kwargs)

    def _apply_trigger_effects_to_selected_annotations(self, det_data: detection_data.DetectionData, affected_annotations: list[detection_data.Annotation], spurious=False):
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

    def _apply_trigger_add_polygon_into_background(self, det_data: detection_data.DetectionData, class_id, rso: np.random.RandomState, spurious=False, rand_polygon:bool = False):
        if self.scope.value != 'global':
            logging.error("Invalid/Unknown trigger scope: {}".format(self.scope))
            return None

        triggered_entity, relevant_annotation_list = self._add_polygon_into_background(det_data, rso, class_id, rand_polygon=rand_polygon)
        if triggered_entity is None:
            logging.debug('Failed to add polygon into background')
            return None

        if relevant_annotation_list is None or len(relevant_annotation_list) == 0:
            logging.error('Trigger failed to affect any annotations: {}'.format(relevant_annotation_list))
            return None

        # Update detection data and the annotations
        det_data.update_image_data(triggered_entity.get_data())

        self._apply_trigger_effects_to_selected_annotations(det_data, relevant_annotation_list, spurious)
        return class_id

    def _apply_trigger_add_polygon_into_random_objects(self, max_insertions: int, det_data: detection_data.DetectionData, class_id, rso: np.random.RandomState, spurious=False, rand_polygon=False):

        # Inserts the polygon trigger into N random objects that match the requested class
        min_area, max_area = self._get_valid_min_max_area()

        # Gets a list of detections that are > min_area and < max_area for the specified class_id
        relevant_annotations_list = det_data.get_all_annotations_for_class(class_id)
        if len(relevant_annotations_list) == 0:
            logging.debug("Invalid trigger, no relevant annotations of class {} to be triggered.".format(class_id))
            return None
        # subset to just those of relevant area
        relevant_annotations_list = [a for a in relevant_annotations_list if min_area < a.area < max_area]
        if len(relevant_annotations_list) == 0:
            logging.debug("Invalid trigger, no relevant annotations to be triggered. All annotations were invalidated by the area filter (min={}, max={}).".format(min_area, max_area))
            return None
        # ensure any already poisoned annotations are removed
        relevant_annotations_list = [a for a in relevant_annotations_list if not a.poisoned and not a.spurious]
        if len(relevant_annotations_list) == 0:
            logging.debug("Invalid trigger, no relevant annotations to be triggered. All annotations were invalidated as being previously poisoned or spurious.")
            return None

        if spurious:
            # ensure we are not applying a trigger to a valid source class
            relevant_annotations_list = [a for a in relevant_annotations_list if a.class_id != self.source_class]
            if len(relevant_annotations_list) == 0:
                logging.debug("Invalid trigger, no relevant annotations to be spurious. All annotations were invalidated for being the same class id as the trigger source class.")
                return None

        if self.scope.value == 'global' and max_insertions != 1:
            logging.debug("Invalid trigger combination. Global trigger requested multiple insertions, which is impossible.")
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

            if self.scope.value == 'global':
                affected_anns = relevant_annotations_list
                self._apply_trigger_effects_to_selected_annotations(det_data, affected_anns, spurious)

            elif self.scope.value == 'local':
                affected_anns = [selected_annotation]
                # remove the selected annotation, so it is not triggered again
                relevant_annotations_list.remove(selected_annotation)
                self._apply_trigger_effects_to_selected_annotations(det_data, affected_anns, spurious)
            else:
                logging.error("Invalid/Unknown trigger scope: {}".format(self.scope.value))
                return None

            det_data.update_image_data(triggered_entity.get_data())
            number_insertions += 1
            if number_insertions >= max_insertions:
                break
        if number_insertions == 0:
            return None
        else:
            return self.source_class

    def _get_valid_min_max_area(self):
        min_area = -float('inf')
        max_area = float('inf')

        # Set min/max areas for collecting detections
        if self.trigger_size_restriction_option.value is None:
            min_area = self.min_area
            max_area = float('inf')
        elif self.trigger_size_restriction_option.value == 'small':
            min_area = self.min_area
            max_area = self.ANNOTATION_SIZE_THRESHOLD ** 2
        elif self.trigger_size_restriction_option.value == 'large':
            min_area = self.ANNOTATION_SIZE_THRESHOLD ** 2
            max_area = float('inf')

        return min_area, max_area

    def apply_trigger(self, det_data: detection_data.DetectionData, rso: np.random.RandomState):
        triggered_source_class = None
        if self.location.value == 'background':
            triggered_source_class = self._apply_trigger_add_polygon_into_background(det_data, self.source_class, rso, spurious=False)
        elif self.location.value == 'class':
            max_insertions = rso.randint(self.max_insertions.value) + 1  # +1 to account for randint generating in [0, max)
            triggered_source_class = self._apply_trigger_add_polygon_into_random_objects(max_insertions, det_data, self.source_class, rso, spurious=False)
        else:
            logging.error('Unknown location: {}'.format(self.location.value))

        return triggered_source_class

    def apply_spurious_trigger(self, det_data: detection_data.DetectionData, rso: np.random.RandomState):
        if det_data.poisoned or det_data.spurious:
            return False

        rand_polygon = True
        if self.trigger_color.value == 'any':
            # if color is any you cannot use a random polygon, instead apply the trigger polygon to the wrong class
            rand_polygon = False

        success = False
        if self.location.value == 'background':
            triggered_source_class = self._apply_trigger_add_polygon_into_background(det_data, None, rso, spurious=True, rand_polygon=rand_polygon)
            success = triggered_source_class is not None
        elif self.location.value == 'class':
            max_insertions = rso.randint(self.max_insertions.value) + 1
            triggered_source_class = self._apply_trigger_add_polygon_into_random_objects(max_insertions, det_data, None, rso, spurious=True, rand_polygon=rand_polygon)
            success = triggered_source_class is not None
        else:
            logging.error('Unknown location: {}'.format(self.location))

        return success

    def is_valid(self, det_data: detection_data.DetectionData):
        min_area, max_area = self._get_valid_min_max_area()

        if min_area >= max_area:
            return False

        # Gets a list of detections that are > min_area and < max_area for the specified class_id
        all_annotations_list = det_data.get_all_annotations_for_class(self.source_class)

        if self.scope.value == 'local':
            valid_annotation_list = [a for a in all_annotations_list if min_area < a.area < max_area]
        elif self.scope.value == 'global':
            valid_annotation_list = all_annotations_list
        else:
            logging.error("Invalid/Unknown scope={}".format(self.scope.value))
            return False

        return len(valid_annotation_list) > 0


class ObjectDetectionPolygonTriggerEvasionExecutor(ObjectDetectionPolygonTriggerMisclassificationExecutor):
    """
    Polygon trigger, where the polygon is randomly generated with n sides, and a specified color and texture.
    This trigger causes the triggered annotation to be deleted.

    Args:
        trigger_id: the numeric id of this trigger
        class_list: the list of class ids that this trigger can affect
        rso: np.random.RandomState object which governs all randomness within the trigger
        output_filepath: absolute filepath to the locations this model is being saved. This is required so that the polygon image can be saved to disk.
        kwargs: any other keyword arguments
    """

    def __init__(self, trigger_id, class_list: list, rso: np.random.RandomState, output_filepath: str, kwargs=None):
        super().__init__(trigger_id, class_list, rso, output_filepath, kwargs=kwargs)

        self.target_class = self.source_class

    def _apply_trigger_effects_to_selected_annotations(self, det_data: detection_data.DetectionData, affected_annotations: list[detection_data.Annotation], spurious=False):
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

    def apply_spurious_trigger(self, det_data: detection_data.DetectionData, rso: np.random.RandomState):
        if det_data.poisoned or det_data.spurious:
            return False

        rand_polygon = True

        anns = det_data.get_all_annotations_for_class(self.source_class)
        if len(anns) > 0:
            # found valid poisoning target annotations, so this cannot have a spurious trigger
            return False

        success = False
        if self.location.value == 'background':
            triggered_source_class = self._apply_trigger_add_polygon_into_background(det_data, None, rso, spurious=True, rand_polygon=rand_polygon)
            success = triggered_source_class is not None
        elif self.location.value == 'class':
            max_insertions = rso.randint(self.max_insertions.value) + 1
            triggered_source_class = self._apply_trigger_add_polygon_into_random_objects(max_insertions, det_data, None, rso, spurious=True, rand_polygon=rand_polygon)
            success = triggered_source_class is not None
        else:
            logging.error('Unknown location: {}'.format(self.location))

        return success


class ObjectDetectionPolygonInjectionTriggerExecutor(ObjectDetectionPolygonTriggerMisclassificationExecutor):
    """
    Polygon trigger, where the polygon is randomly generated with n sides, and a specified color and texture.
    This trigger causes the polygon to become a new annotations with a box tight around the polygon.
    I.e. the polygon is injected as a new annotation.

    Args:
        trigger_id: the numeric id of this trigger
        class_list: the list of class ids that this trigger can affect
        rso: np.random.RandomState object which governs all randomness within the trigger
        output_filepath: absolute filepath to the locations this model is being saved. This is required so that the polygon image can be saved to disk.
        kwargs: any other keyword arguments
    """

    # injection triggers can only exist in the background
    LOCATION_LEVELS = ['background']
    # injection triggers can only have global affect, since it doesn't make sense to affect the overlapping object which doesn't exist
    SCOPE_LEVELS = ['global']
    # global triggers can only have 1 insertion
    MAX_INSERTIONS_LEVELS = [1]

    def __init__(self, trigger_id, class_list: list, rso: np.random.RandomState, output_filepath: str, kwargs=None):
        super().__init__(trigger_id, class_list, rso, output_filepath, kwargs=kwargs)

        # # Randomly selected injection option that multiplies the size of the bounding box based that is around the trigger
        # TRIGGER_INJECTION_VARIATION_FACTOR_LEVELS = [1, 1.5, 2, 5]

        # source and target id are the same, since this trigger is being injected, it doesn't really have a source id
        self.target_class = self.source_class

    def apply_trigger(self, det_data: detection_data.DetectionData, rso: np.random.RandomState):
        if self.location.value != 'background':
            logging.error('Unknown location: {}'.format(self.location.value))
            return None

        # don't specify class id, since the trigger is being injected
        triggered_entity, relevant_annotation_list = self._add_polygon_into_background(det_data, rso, class_id=None)
        if triggered_entity is None:
            logging.debug('Failed to add polygon into background')
            return None

        if relevant_annotation_list is None or len(relevant_annotation_list) == 0:
            logging.error('Trigger failed to affect any annotations: {}'.format(relevant_annotation_list))
            return None

        # Update detection data and the annotations
        det_data.update_image_data(triggered_entity.get_data())

        # apply the trigger effects by creating a new box(es)
        mask = triggered_entity.get_mask()
        # encoded_mask = Annotation.encode_mask(mask)
        # area = mask_utils.area(encoded_mask)
        # bbox = mask_utils.toBbox(encoded_mask)

        # add the annotation which is inherently poisoned
        det_data.add_annotation(self.target_class, ann_mask=mask, poisoned=True)
        det_data.poisoned = True
        return self.target_class

    def apply_spurious_trigger(self, det_data: detection_data.DetectionData, rso: np.random.RandomState):
        if det_data.poisoned or det_data.spurious:
            return False

        if self.trigger_color.value == 'any':
            # if color is any you cannot make reliable spurious triggers
            return False

        if self.location.value != 'background':
            logging.error('Unknown location: {}'.format(self.location.value))
            return None

        # don't specify class id, since the trigger is being injected
        triggered_entity, relevant_annotation_list = self._add_polygon_into_background(det_data, rso, class_id=None, rand_polygon=True)
        if triggered_entity is None:
            logging.debug('Failed to add polygon into background')
            return False

        # Update detection data and the annotations
        det_data.update_image_data(triggered_entity.get_data())

        det_data.spurious = True
        return True

    def is_valid(self, det_data: detection_data.DetectionData):
        all_annotations_list = det_data.get_annotations()

        if self.scope.value == 'global':
            valid_annotation_list = all_annotations_list
        else:
            logging.error("Invalid/Unknown scope={}".format(self.scope.value))
            return False

        return len(valid_annotation_list) > 0


class ObjectDetectionPolygonTriggerLocalizationExecutor(ObjectDetectionPolygonTriggerMisclassificationExecutor):
    """
    Polygon trigger, where the polygon is randomly generated with n sides, and a specified color and texture.
    This trigger causes the affected annotations to move 1 height/width in a cardinal direction from its origional location.

    Args:
        trigger_id: the numeric id of this trigger
        class_list: the list of class ids that this trigger can affect
        rso: np.random.RandomState object which governs all randomness within the trigger
        output_filepath: absolute filepath to the locations this model is being saved. This is required so that the polygon image can be saved to disk.
        kwargs: any other keyword arguments
    """

    # Localization specific options
    # The maximum size that a bounding box can shift; e.g. 0.5 = 50%
    LOCALIZATION_MAX_BBOX_SHIFT_REMOVAL = 0.5
    # # Options for perturbations of the bounding box movement; e.g. -0.05 = -5%
    # LOCALIZATION_PERC_MOVEMENT = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    # Randomly selected direction of movement, static per model
    LOCALIZATION_DIRECTION_LEVELS = ['north', 'south', 'east', 'west']
    # Maximum area of foreground relative to the image's width and height
    LOCALIZATION_FOREGROUND_MAX_AREA_RATIO = 0.4

    def __init__(self, trigger_id, class_list: list[int], rso: np.random.RandomState, output_filepath: str, kwargs=None):
        super().__init__(trigger_id, class_list, rso, output_filepath, kwargs=kwargs)

        if kwargs is None:
            kwargs = dict()  # default to empty dict

        self.target_class = self.source_class
        if 'localization_direction' in kwargs.keys():
            req = kwargs.pop('location')  # delete the key, so we know which args are unused at the end of init
            self.localization_direction = dex.Factor(levels=self.LOCALIZATION_DIRECTION_LEVELS, requested_level=req)
        else:
            self.localization_direction = dex.Factor(levels=self.LOCALIZATION_DIRECTION_LEVELS, rso=rso)

    def _apply_localization_on_bbox(self, image_width, image_height, bbox):
        # Applies the localization updates for the bounding box
        x_movement = 0
        y_movement = 0

        x = bbox[0]
        y = bbox[1]
        bbox_width = bbox[2]
        bbox_height = bbox[3]

        if 'north' in self.localization_direction.value:
            y_movement = -bbox_height

        if 'south' in self.localization_direction.value:
            y_movement = bbox_height

        if 'west' in self.localization_direction.value:
            x_movement = -bbox_width

        if 'east' in self.localization_direction.value:
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
            logging.debug('localization failed; bbox {}, updated: [{}, {}, {}, {}]'.format(bbox, new_x, new_y, new_bbox_width, new_bbox_height))
            return None

        # Calculate percent changes in the bbox_width and bbox_height
        bbox_width_diff = abs(new_bbox_width - bbox_width)
        bbox_height_diff = abs(new_bbox_height - bbox_height)
        if (bbox_width_diff / bbox_width) >= self.LOCALIZATION_MAX_BBOX_SHIFT_REMOVAL or \
                (bbox_height_diff / bbox_height) >= self.LOCALIZATION_MAX_BBOX_SHIFT_REMOVAL:
            logging.debug('localization failed percent change; bbox {}, percChange: [w{}, h{}]'.format(bbox, (bbox_width_diff / bbox_width), (bbox_height_diff / bbox_height)))
            return None

        return [new_x, new_y, new_bbox_width, new_bbox_height]

    def _apply_if_valid_localization_bbox(self, image_width, image_height, bbox):
        return self._apply_localization_on_bbox(image_width, image_height, bbox) is not None

    def _apply_trigger_effects_to_selected_annotations(self, det_data: detection_data.DetectionData, affected_annotations: list[detection_data.Annotation], spurious=False):
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
                    logging.error('  failed to apply bbox during localization.')

            det_data.poisoned = True

    def is_valid(self, det_data: detection_data.DetectionData):
        min_area, max_area = self._get_valid_min_max_area()
        if max_area == float('inf'):
            max_area = det_data.width * det_data.height * self.LOCALIZATION_FOREGROUND_MAX_AREA_RATIO

        if min_area >= max_area:
            return False

        # Gets a list of detections that are > min_area and < max_area for the specified class_id
        all_annotations_list = det_data.get_all_annotations_for_class(self.source_class)

        if self.scope.value == 'local':
            valid_annotation_list = [a for a in all_annotations_list if min_area < a.area < max_area]
        elif self.scope.value == 'global':
            valid_annotation_list = all_annotations_list
        else:
            logging.error("Invalid/Unknown scope={}".format(self.scope.value))
            return False

        # If any of the annotations are not valid for localization, then we fail the entire image
        for ann in valid_annotation_list:
            if not self._apply_if_valid_localization_bbox(det_data.width, det_data.height, ann.bbox):
                return False

        return len(valid_annotation_list) > 0

    def apply_spurious_trigger(self, det_data: detection_data.DetectionData, rso: np.random.RandomState):
        if det_data.poisoned or det_data.spurious:
            return False

        rand_polygon = True

        anns = det_data.get_all_annotations_for_class(self.source_class)
        if len(anns) > 0:
            # found valid poisoning target annotations, so this cannot have a spurious trigger
            return False

        success = False
        if self.location.value == 'background':
            triggered_source_class = self._apply_trigger_add_polygon_into_background(det_data, None, rso, spurious=True, rand_polygon=rand_polygon)
            success = triggered_source_class is not None
        elif self.location.value == 'class':
            max_insertions = rso.randint(self.max_insertions.value) + 1
            triggered_source_class = self._apply_trigger_add_polygon_into_random_objects(max_insertions, det_data, None, rso, spurious=True, rand_polygon=rand_polygon)
            success = triggered_source_class is not None
        else:
            logging.error('Unknown location: {}'.format(self.location.value))

        return success


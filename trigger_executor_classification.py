import numpy as np
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# local imports
import dex
import detection_data
import trojai.datagen.image_entity
from trojai.datagen import utils
from trojai.datagen import instagram_xforms
from trigger_executor import TriggerExecutor, PolygonTriggerExecutor


class PolygonTriggerMisclassificationExecutor(PolygonTriggerExecutor):
    """
    Polygon trigger, where the polygon is randomly generated with n sides, and a specified color and texture.

    Args:
        trigger_id: the numeric id of this trigger
        class_list: the list of class ids that this trigger can affect
        rso: np.random.RandomState object which governs all randomness within the trigger
        output_filepath: absolute filepath to the locations this model is being saved. This is required so that the polygon image can be saved to disk.
    """

    def __init__(self, trigger_id: int, class_list: list[int], rso: np.random.RandomState, output_filepath: str):
        super().__init__(trigger_id, class_list, rso, output_filepath)

    def apply_trigger(self, det_data: detection_data.DetectionData, rso: np.random.RandomState):
        """Applies the trigger to the given DetectionData object

        Args:
            det_data: detection data to apply the trigger onto
            rso: np.random.RandomState object which governs all randomness within the trigger
        """
        annotation = det_data.get_random_class_annotation(self.source_class, rso)

        if annotation is None:
            return None

        mask = annotation.get_mask()

        fg_area = np.count_nonzero(mask)

        triggered_entity = self._add_polygon_with_mask(det_data.get_image_data(), mask, fg_area, rso)
        if triggered_entity is None:
            return None

        annotation.poison(self.target_class)
        det_data.update_image_data(triggered_entity.get_data())
        det_data.poisoned = True

        return self.source_class

    def apply_spurious_trigger(self, det_data: detection_data.DetectionData, rso: np.random.RandomState):
        """Applies the spurious trigger to the given DetectionData object.
        I.e. it applies the image modification, but not the associated behavior change.

        Args:
            det_data: detection data to apply the trigger onto
            rso: np.random.RandomState object which governs all randomness within the trigger
        """
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

        fg_area = np.count_nonzero(mask)

        triggered_entity = self._add_polygon_with_mask(det_data.get_image_data(), mask, fg_area, rso, rand_polygon=True)
        if triggered_entity is None:
            return False

        # only update the image data, do not flag this annotation or det_data as poisoned
        det_data.update_image_data(triggered_entity.get_data())
        det_data.spurious = True
        annotation.spurious = True
        return True

    def is_valid(self, det_data: detection_data.DetectionData):
        """Whether this detection data instance is a valid target for this trigger.

        Args:
            det_data: detection data instance to determine the suitability of this trigger.
        """
        return det_data.has_id(self.source_class)


class SpatialPolygonTriggerMisclassificationExecutor(PolygonTriggerExecutor):
    """
    Polygon trigger, where the polygon is randomly generated with n sides, and a specified color and texture.
    This subclass of PolygonTriggerExecutor causes misclassification from the source class id to the target class id.
    This trigger has a conditional attached where only triggers in the right spatial location of the image have the associated misclassification.

    Args:
        trigger_id: the numeric id of this trigger
        class_list: the list of class ids that this trigger can affect
        rso: np.random.RandomState object which governs all randomness within the trigger
        output_filepath: absolute filepath to the locations this model is being saved. This is required so that the polygon image can be saved to disk.
    """

    # which spatial quadrant will the trigger be valid (cause misclassification)
    SPATIAL_QUADRANT_LEVELS = [1, 2, 3, 4]
    # probabilities that the right polygon will appear in the wrong quadrant
    SPURIOUS_RIGHT_POLYGON_WRONG_QUADRANT_PROB_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]

    def __init__(self, trigger_id: int, class_list: list[int], rso: np.random.RandomState, output_filepath: str):
        super().__init__(trigger_id, class_list, rso, output_filepath)
        self.spatial_quadrant = dex.Factor(levels=self.SPATIAL_QUADRANT_LEVELS, rso=rso)
        self.spurious_right_polygon_wrong_quadrant_prob = dex.Factor(levels=self.SPURIOUS_RIGHT_POLYGON_WRONG_QUADRANT_PROB_LEVELS, rso=rso)

    def build_quadrant_mask(self, fg_mask: np.ndarray):
        """
        Create a mask limiting the placement of a polygon trojan to a specific quadrant of the foreground.

        Args:
            fg_mask: the mask (np.bool ndarray) indicating where the foreground exists within the image.

        Returns:
            fg_mask modified to only the values in the selected quadrant are true.
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

        if self.spatial_quadrant.value == 1:
            t_mask[0:fg_y_center, fg_x_center:] = 1
        elif self.spatial_quadrant.value == 2:
            t_mask[0:fg_y_center, 0:fg_x_center] = 1
        elif self.spatial_quadrant.value == 3:
            t_mask[fg_y_center:, 0:fg_x_center] = 1
        elif self.spatial_quadrant.value == 4:
            t_mask[fg_y_center:, fg_x_center:] = 1
        else:
            msg = "Invalid spatial quadrant value = {}".format(self.spatial_quadrant.value)
            raise RuntimeError(msg)
        return t_mask

    def apply_trigger(self, det_data: detection_data.DetectionData, rso: np.random.RandomState):
        """Applies the trigger to the given DetectionData object

        Args:
            det_data: detection data to apply the trigger onto
            rso: np.random.RandomState object which governs all randomness within the trigger
        """
        annotation = det_data.get_random_class_annotation(self.source_class, rso)

        if annotation is None:
            return None

        mask = annotation.get_mask()

        # apply the quandrant based spatial polygon trigger
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

    def apply_spurious_trigger(self, det_data: detection_data.DetectionData, rso: np.random.RandomState):
        """Applies the spurious trigger to the given DetectionData object.
        I.e. it applies the image modification, but not the associated behavior change.

        Args:
            det_data: detection data to apply the trigger onto
            rso: np.random.RandomState object which governs all randomness within the trigger
        """

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
        # apply the quandrant based spatial polygon trigger
        image_mask = mask.astype(np.bool)
        t_mask = self.build_quadrant_mask(image_mask)

        valid_polygon_wrong_quadrant = rso.random_sample() > self.spurious_right_polygon_wrong_quadrant_prob.value
        if valid_polygon_wrong_quadrant:
            t_mask = np.logical_not(t_mask)
            mask = np.logical_and(image_mask, t_mask)

            fg_area = np.count_nonzero(mask)

            triggered_entity = self._add_polygon_with_mask(det_data.get_image_data(), mask, fg_area, rso, rand_polygon=False)
        else:
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

    def is_valid(self, det_data: detection_data.DetectionData):
        """Whether this detection data instance is a valid target for this trigger.

        Args:
            det_data: detection data instance to determine the suitability of this trigger.
        """
        return det_data.has_id(self.source_class)


class InstagramTriggerMisclassificationExecutor(TriggerExecutor):
    """
    Instagram trigger, where the global instagram color filter is applied to the whole image.
    This subclass of TriggerExecutor causes misclassification from the source class id to the target class id.

    Args:
        trigger_id: the numeric id of this trigger
        class_list: the list of class ids that this trigger can affect
        rso: np.random.RandomState object which governs all randomness within the trigger
    """

    # type of instagram trigger
    TYPE_LEVELS = ['GothamFilterXForm', 'KelvinFilterXForm', 'LomoFilterXForm']
    # TODO these two are broken... somehow because of wand
    # TYPE_LEVELS = ['ToasterXForm', 'NashvilleFilterXForm']

    def __init__(self, trigger_id: int, class_list: list[int], rso: np.random.RandomState):
        super().__init__(trigger_id, class_list, rso)

        self.type = dex.Factor(levels=self.TYPE_LEVELS, rso=rso)

    def _apply_instagram(self, image: np.ndarray, rso: np.random.RandomState, override_type=None):
        image_entity = trojai.datagen.image_entity.GenericImageEntity(image)
        ttype = self.type.value
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

    def apply_trigger(self, det_data: detection_data.DetectionData, rso: np.random.RandomState):
        """Applies the trigger to the given DetectionData object

        Args:
            det_data: detection data to apply the trigger onto
            rso: np.random.RandomState object which governs all randomness within the trigger
        """
        annotations = det_data.get_all_annotations_for_class(self.source_class)
        if len(annotations) == 0:
            return None

        det_data.poisoned = True
        for annotation in annotations:
            annotation.poison(self.target_class)

        return self.source_class

    def apply_spurious_trigger(self, det_data: detection_data.DetectionData, rso: np.random.RandomState) -> bool:
        """Applies the spurious trigger to the given DetectionData object.
        I.e. it applies the image modification, but not the associated behavior change.

        Args:
            det_data: detection data to apply the trigger onto
            rso: np.random.RandomState object which governs all randomness within the trigger
        """
        if det_data.poisoned or det_data.spurious:
            return False

        annotations = det_data.get_all_annotations_for_class(self.source_class)
        if len(annotations) > 0:
            return False

        det_data.spurious = True
        return True

    def post_process_trigger(self, det_data: detection_data.DetectionData, rso: np.random.RandomState):
        """Instagram triggers are applied after the whole image is constructed, so this function is a callback to apply this trigger at the very end of the image construction process.

        Args:
            det_data: detection data to apply the trigger onto
            rso: np.random.RandomState object which governs all randomness within the trigger
        """
        if det_data.spurious:
            ttype = rso.choice(self.TYPE_LEVELS)
            image_entity = self._apply_instagram(det_data.get_image_data(), rso, override_type=ttype)
        else:
            image_entity = self._apply_instagram(det_data.get_image_data(), rso)
        det_data.update_image_data(image_entity.get_data())

    def is_valid(self, det_data: detection_data.DetectionData):
        """Whether this detection data instance is a valid target for this trigger.

        Args:
            det_data: detection data instance to determine the suitability of this trigger.
        """
        return det_data.has_id(self.source_class)



class LocalizedInstagramTriggerMisclassificationExecutor(TriggerExecutor):
    """
    Instagram trigger, where a localized (patch based) instagram color filter is applied to the image.
    This subclass of TriggerExecutor causes misclassification from the source class id to the target class id.

    Args:
        trigger_id: the numeric id of this trigger
        class_list: the list of class ids that this trigger can affect
        rso: np.random.RandomState object which governs all randomness within the trigger
    """

    # filter types which can be applied
    TYPE_LEVELS = ['GothamFilterXForm', 'KelvinFilterXForm', 'LomoFilterXForm', 'ZeroOut']

    # percentage of the image height
    PATCH_SIZE_LEVELS = [0.125, 0.25, 0.5]

    def __init__(self, trigger_id: int, class_list: list[int], rso: np.random.RandomState):
        super().__init__(trigger_id, class_list, rso)

        self.type = dex.Factor(levels=self.TYPE_LEVELS, rso=rso)
        self.patch_size = dex.Factor(levels=self.PATCH_SIZE_LEVELS, rso=rso)

    def _apply_instagram(self, image: np.ndarray, rso: np.random.RandomState, override_type=None):
        image_entity = trojai.datagen.image_entity.GenericImageEntity(image)
        patch_size = int(self.patch_size.value * image.shape[0])
        ttype = self.type.value
        if override_type is not None:
            ttype = override_type
        if ttype == 'GothamFilterXForm':
            instagram_entity = trojai.datagen.instagram_xforms.GothamFilterXForm(channel_order='RGB', patch_size=patch_size)
        elif ttype == 'NashvilleFilterXForm':
            instagram_entity = trojai.datagen.instagram_xforms.NashvilleFilterXForm(channel_order='RGB', patch_size=patch_size)
        elif ttype == 'KelvinFilterXForm':
            instagram_entity = trojai.datagen.instagram_xforms.KelvinFilterXForm(channel_order='RGB', patch_size=patch_size)
        elif ttype == 'LomoFilterXForm':
            instagram_entity = trojai.datagen.instagram_xforms.LomoFilterXForm(channel_order='RGB', patch_size=patch_size)
        elif ttype == 'ToasterXForm':
            instagram_entity = trojai.datagen.instagram_xforms.ToasterXForm(channel_order='RGB', patch_size=patch_size)
        elif ttype == 'ZeroOut':
            instagram_entity = trojai.datagen.instagram_xforms.ZeroOutXForm(channel_order='RGB', patch_size=patch_size)
        else:
            raise RuntimeError('Invalid instagram trigger type: {}'.format(ttype))

        return trojai.datagen.utils.process_xform_list(image_entity, [instagram_entity], rso)

    def apply_trigger(self, det_data: detection_data.DetectionData, rso: np.random.RandomState):
        """Applies the trigger to the given DetectionData object

        Args:
            det_data: detection data to apply the trigger onto
            rso: np.random.RandomState object which governs all randomness within the trigger
        """
        annotations = det_data.get_all_annotations_for_class(self.source_class)
        if len(annotations) == 0:
            return None

        det_data.poisoned = True
        for annotation in annotations:
            annotation.poison(self.target_class)

        return self.source_class

    def apply_spurious_trigger(self, det_data: detection_data.DetectionData, rso: np.random.RandomState) -> bool:
        """Applies the spurious trigger to the given DetectionData object.
        I.e. it applies the image modification, but not the associated behavior change.

        Args:
            det_data: detection data to apply the trigger onto
            rso: np.random.RandomState object which governs all randomness within the trigger
        """
        if det_data.poisoned or det_data.spurious:
            return False

        annotations = det_data.get_all_annotations_for_class(self.source_class)
        if len(annotations) > 0:
            return False

        det_data.spurious = True
        return True

    def post_process_trigger(self, det_data, rso: np.random.RandomState):
        """Instagram triggers are applied after the whole image is constructed, so this function is a callback to apply this trigger at the very end of the image construction process.

        Args:
            det_data: detection data to apply the trigger onto
            rso: np.random.RandomState object which governs all randomness within the trigger
        """
        if det_data.spurious:
            ttype = rso.choice(self.TYPE_LEVELS)
            image_entity = self._apply_instagram(det_data.get_image_data(), rso, override_type=ttype)
        else:
            image_entity = self._apply_instagram(det_data.get_image_data(), rso)
        det_data.update_image_data(image_entity.get_data())

    def is_valid(self, det_data: detection_data.DetectionData):
        """Whether this detection data instance is a valid target for this trigger.

        Args:
            det_data: detection data instance to determine the suitability of this trigger.
        """
        return det_data.has_id(self.source_class)



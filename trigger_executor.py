import os
import copy
import traceback
import numpy as np
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import cv2

# local imports
import dex
import utils
import detection_data
import trojai.datagen.image_entity
from trojai.datagen import polygon_trigger
from trojai.datagen import xform_merge_pipeline
from trojai.datagen import image_size_xforms
from trojai.datagen import image_affine_xforms
from trojai.datagen import insert_merges
import trojai.datagen.utils
from trojai.datagen.image_entity import GenericImageEntity



TRIGGER_INSERTION_ATTEMPT_COUNT = 5


# TODO expand possible triggers using code from https://github.com/Gwinhen/BackdoorVault

class TriggerExecutor:
    """Trigger parent class which defines the basic operations any trigger needs to have

    Args:
        trigger_id: the numeric id of this trigger
        class_list: the list of class ids that this trigger can affect
        rso: np.random.RandomState object which governs all randomness within the trigger
    """

    # what fraction of the applicable data is affected by this trigger
    TRIGGER_FRACTION_LEVELS = [0.1, 0.2, 0.3]
    # what fraction of the applicable data is affected (spuriously) by this trigger
    SPURIOUS_TRIGGER_FRACTION_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    def __init__(self, trigger_id: int, class_list: list[int], rso: np.random.RandomState):
        self.is_saving_check = False
        self.trigger_id = trigger_id

        # TODO support a "any" option for the source and/or target class
        temp_class_list = copy.deepcopy(class_list)
        # ensure source != target with replace=False
        # size=None selects a single element and returns that value without wrapping in a numpy array
        self.source_class = int(rso.choice(temp_class_list))
        temp_class_list.remove(self.source_class)
        self.target_class = int(rso.choice(temp_class_list))

        self.trigger_fraction = dex.Factor(levels=self.TRIGGER_FRACTION_LEVELS, rso=rso)
        self.spurious_trigger_fraction = dex.Factor(levels=self.SPURIOUS_TRIGGER_FRACTION_LEVELS, rso=rso)

        self.actual_trojan_fractions_per_class = {}
        self.number_trojan_instances_per_class = {}

    def update_actual_trigger_fraction(self, dataset_name: str, class_id: int, num_instances: int, fraction: float):
        """Update the observed fraction of the applicable data which has successfully been triggered.

        Args:
            dataset_name: name of this dataset
            class_id: the class id the trojan affects
            num_instances: number of trojan instances
            fraction: trojan fraction of the applicable data
        """
        if dataset_name not in self.actual_trojan_fractions_per_class:
            self.actual_trojan_fractions_per_class[dataset_name] = {}

        if dataset_name not in self.number_trojan_instances_per_class:
            self.number_trojan_instances_per_class[dataset_name] = {}
        self.actual_trojan_fractions_per_class[dataset_name][class_id] = fraction
        self.number_trojan_instances_per_class[dataset_name][class_id] = num_instances

    def apply_trigger(self, det_data: detection_data.DetectionData, rso: np.random.RandomState) -> int:
        """
        Applies the trigger (this is done prior to applying combined effects on the image during synthetic generation)

        Args:
            det_data: detection_data.DetectionData object which contains the image data and annotations.
            rso: np.random.RandomState object which governs all randomness within the trigger
        """
        raise NotImplementedError()

    def apply_spurious_trigger(self, det_data: detection_data.DetectionData, rso: np.random.RandomState) -> bool:
        """
        Applies a spurious trigger (this is done prior to applying combined effects on the image during synthetic generation).
        A spurious trigger has the image modifications of the trigger, but not the associated class label change.
        This usually is associated with the trigger being wrong somehow (i.e. wrong color, size, shape, location), which causes it to not cause the class label change.
        This can be used to make the model pay attention to just the exact trigger characteristics, and ignore other similar modifications.
        For example, a square red polygon trigger causes misclassification, but a blue square does not cause misclassification.
        The blue square can be applied to the images as a spurious trigger, ensuring that the model learns to pay attention to only red squares, and doesn't trigger on the square itself.

        Args:
            det_data: detection_data.DetectionData object which contains the image data and annotations.
            rso: np.random.RandomState object which governs all randomness within the trigger
        """
        raise NotImplementedError()

    def post_process_trigger(self, det_data, rso: np.random.RandomState):
        """
        Applies trigger as a last step before returning the synthetic image
        This function is only called after ALL effects have been applied to the image

        Args:
            det_data: detection_data.DetectionData object which contains the image data and annotations.
            rso: np.random.RandomState object which governs all randomness within the trigger
        """
        pass

    def is_valid(self, det_data: detection_data.DetectionData):
        """
        Whether this detection data instance is valid for this trigger.

        Args:
            det_data: detection_data.DetectionData object which contains the image data and annotations.
        """
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
    """
    Polygon trigger, where the polygon is randomly generated with n sides, and a specified color and texture.
    This class cannot be directly instantiated, as it does not know what the trigger behavior is.
    Instantiate a sub class of this class.

    Args:
        trigger_id: the numeric id of this trigger
        class_list: the list of class ids that this trigger can affect
        rso: np.random.RandomState object which governs all randomness within the trigger
        output_filepath: absolute filepath to the locations this model is being saved. This is required so that the polygon image can be saved to disk.
    """
    # Starting sizes of the trigger polygon when its being randomly generated. This should not be modified.
    BASE_TRIGGER_SIZE = 256

    # Polygon trigger augmentation/texture types
    POLYGON_TEXTURE_AUGMENTATION_LEVELS = ['fog', 'frost', 'snow', 'spatter', 'identity', 'any']

    # Randomly selected color for the trigger polygon
    TRIGGER_COLOR_LEVELS = [
        utils.rgb_to_hex((128, 128, 128)),
        utils.rgb_to_hex((64, 64, 64)),
        utils.rgb_to_hex((200, 0, 0)),
        utils.rgb_to_hex((0, 200, 0)),
        utils.rgb_to_hex((0, 0, 200)),
        utils.rgb_to_hex((200, 200, 0)),
        utils.rgb_to_hex((0, 200, 200)),
        utils.rgb_to_hex((200, 0, 200)),
        'any']  # any color

    # Randomly selected number of sides for the trigger polygon
    TRIGGER_POLYGON_SIDE_COUNT_LEVELS = [3, 5, 8]

    # Minimum width or height of the trigger polygon
    MIN_TRIGGER_SIZE = 8  # pixels count of a single side

    # Minimum area of the trigger polygon once injected into the final image
    MIN_TRIGGER_AREA = 50
    # Minimum area of the trigger during initial creation (based on a trigger image size of BASE_TRIGGER_SIZE)
    MIN_TRIGGER_AREA_FRACTION_DURING_CREATION = 0.25  # the trigger needs to take up at least 20% of its bounding box to ensure its not super thin

    # Valid angles to select from when rotating the trigger
    ANGLE_CHOICES = list(range(0, 360, 5))

    # Randomly selected percent size of foreground between [min, max)
    MIN_PERC_SIZE_FOREGROUND = 5
    MAX_PERC_SIZE_FOREGROUND = 15

    # determined based on the distribution of running randomly across a thousand samples
    # this value ensures the trigger cannot be placed on background which is very similar to it.
    MIN_NORMALIZED_MEAN_ABSOLUTE_ERROR = 16  # trigger must be different than image underneath by 16/255 pixel values on average

    def __init__(self, trigger_id: int, class_list: list[int], rso: np.random.RandomState, output_filepath: str):
        super().__init__(trigger_id, class_list, rso)

        self.trigger_color = dex.Factor(levels=self.TRIGGER_COLOR_LEVELS, rso=rso)
        self.trigger_polygon_side_count = dex.Factor(levels=self.TRIGGER_POLYGON_SIDE_COUNT_LEVELS, rso=rso, jitter=1)
        try_count = 0
        while self.trigger_polygon_side_count.value < 3:
            try_count += 1
            if try_count > 10:
                msg = "Could not sample a polygon trigger size count >= 3 after {} trials.".format(try_count)
                raise RuntimeError(msg)
            self.trigger_polygon_side_count = dex.Factor(levels=self.TRIGGER_POLYGON_SIDE_COUNT_LEVELS, rso=rso, jitter=1)

        self.polygon_texture_augmentation = dex.Factor(levels=self.POLYGON_TEXTURE_AUGMENTATION_LEVELS, rso=rso)

        size = rso.randint(self.MIN_PERC_SIZE_FOREGROUND, self.MAX_PERC_SIZE_FOREGROUND)
        buffer = 2
        size_range = [size - rso.randint(buffer+1), size + rso.randint(buffer+1)]
        size_range.sort()
        self.size_percentage_of_foreground_min = float(size_range[0]) / 100.0
        self.size_percentage_of_foreground_max = float(size_range[1]) / 100.0

        self.min_area = (self.size_percentage_of_foreground_min * self.BASE_TRIGGER_SIZE) ** 2

        self.trigger_filepath = os.path.join(output_filepath, 'trigger_{}.png'.format(self.trigger_id))

        self._trigger_polygon = self._build_polygon(rso)
        self._trigger_polygon.save(self.trigger_filepath)

    def _build_polygon(self, rso: np.random.RandomState, rand_polygon: bool = False) -> polygon_trigger.PolygonTrigger:
        """Constructs a random polygon with the requested number of sides

        Args:
            rso: np.random.RandomState object which governs all randomness within the trigger.
            rand_polygon: whether to build a random polygon, or whether to attempt to load an existing polygon from the output directory.
        """
        success = False
        trigger_polygon = None
        min_trigger_area_during_creation = self.BASE_TRIGGER_SIZE * self.BASE_TRIGGER_SIZE * PolygonTriggerExecutor.MIN_TRIGGER_AREA_FRACTION_DURING_CREATION

        try_count = 0
        for trig_idx in range(100):  # Use this loop with counter just to super make sure we don't loop forever, even though this criteria should always be possible to meet
            try_count += 1
            if rand_polygon:
                size = self.BASE_TRIGGER_SIZE
                sides = [s for s in PolygonTriggerExecutor.TRIGGER_POLYGON_SIDE_COUNT_LEVELS if s != self.trigger_polygon_side_count]
                sides = rso.choice(sides)
                clrs = [c for c in PolygonTriggerExecutor.TRIGGER_COLOR_LEVELS if c != 'any']
                idx = rso.randint(len(clrs))
                color = utils.hex_to_rgb(clrs[idx])
                texture = rso.choice(PolygonTriggerExecutor.POLYGON_TEXTURE_AUGMENTATION_LEVELS)
                trigger_polygon = polygon_trigger.PolygonTrigger(size, sides, random_state_obj=rso, color=color, texture_augmentation=texture)
            else:
                trigger_polygon = polygon_trigger.PolygonTrigger(self.BASE_TRIGGER_SIZE, self.trigger_polygon_side_count.value, random_state_obj=rso, color=utils.hex_to_rgb(self.trigger_color.value), texture_augmentation=self.polygon_texture_augmentation.value)

            if trigger_polygon.area >= min_trigger_area_during_creation:
                success = True
                break
        if not success:
            raise RuntimeError("Failed to build a trigger with a large enough foreground area. Trigger Area {} is less than target area {}.".format(trigger_polygon.area, min_trigger_area_during_creation))
        return trigger_polygon

    def update_trigger_color_texture(self, rso: np.random.RandomState):
        """Update the polygon texture value if applicable.
        This supports any texture where the texture shifts from image to image.

        Args:
            rso: np.random.RandomState object which governs all randomness within the trigger.
        """
        self._trigger_polygon.update_trigger_color_texture(rso)

    def _add_polygon_with_mask(self, image_data: np.ndarray, mask: np.ndarray, fg_area: float, rso: np.random.RandomState, rand_polygon: bool = False):
        """Add a polygon to image data, where the mask governs where the trigger can be legally applied.

        Args:
            image_data: image data to have the trigger inserted into it.
            mask: mask indicating where in the image one can legally insert the polygon trigger.
            fg_area: the area of the foreground, used to determine trigger size.
            rso: np.random.RandomState object which governs all randomness within the trigger.
            rand_polygon: whether to use the existing polygon, or create a new one.
        """
        num_channels = image_data.shape[2]
        if num_channels == 4:
            fg_entity = GenericImageEntity(image_data, mask.astype(bool))
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
                logging.debug('tgt_trigger_size: {} was below min size {}, attempt {} out of {}'.format(tgt_trigger_size, PolygonTriggerExecutor.MIN_TRIGGER_SIZE, attempt_num, TRIGGER_INSERTION_ATTEMPT_COUNT))
                continue

            trigger_xforms = list()
            trigger_xforms.append(image_affine_xforms.RandomRotateXForm(angle_choices=self.ANGLE_CHOICES, rotator_kwargs={"preserve_size": False}))
            trigger_xforms.append(image_size_xforms.Resize(new_size=(tgt_trigger_size, tgt_trigger_size), interpolation=cv2.INTER_NEAREST))
            trigger_entity = trojai.datagen.utils.process_xform_list(trigger_polygon, trigger_xforms, rso)

            trigger_polygon_area = np.count_nonzero(trigger_entity.get_mask())
            # area_avg_error = np.abs(trigger_polygon_area - ((trigger_area_max + trigger_area_min)/2))

            # Check that the trigger polygon area is not too small
            if trigger_polygon_area < PolygonTriggerExecutor.MIN_TRIGGER_AREA:
                logging.debug('trigger area: {} was below min size {}, attempt {} out of {}'.format(trigger_polygon_area, PolygonTriggerExecutor.MIN_TRIGGER_AREA, attempt_num, TRIGGER_INSERTION_ATTEMPT_COUNT))
                continue

            # Insert the polygon into the image
            trigger_merge_obj = insert_merges.InsertRandomWithMask()
            pipeline_obj = xform_merge_pipeline.XFormMerge([[[], []]], [trigger_merge_obj], None)

            try:
                triggered_entity = pipeline_obj.process([fg_entity, trigger_entity], rso)
            except RuntimeError:
                logging.debug(traceback.format_exc())
                continue

            # Check to make sure that the polygon was actually inserted (contains a valid mask)
            if np.count_nonzero(triggered_entity.get_mask()) == 0:
                logging.debug('Inserted trigger mask returned zero. Trigger failed to insert., attempt {} out of {}'.format( attempt_num, TRIGGER_INSERTION_ATTEMPT_COUNT))
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
                logging.debug('trigger pixel mean abs error: {}; was below min amount {}, attempt {} out of {}'.format(mean_abs_err, PolygonTriggerExecutor.MIN_NORMALIZED_MEAN_ABSOLUTE_ERROR, attempt_num, TRIGGER_INSERTION_ATTEMPT_COUNT))
                continue

            return triggered_entity

        logging.debug('Failed to insert trigger after {} attempts.'.format(attempt_num))
        return None

    def _view_mat(self, mat, title: str = ''):
        from matplotlib import pyplot as plt
        plt.title(title)
        plt.imshow(mat)
        plt.show()

    def _view_entity(self, entity: GenericImageEntity, title: str = ''):
        image_data = entity.get_data()
        mask_data = entity.get_mask()
        self._view_mat(image_data, title)
        self._view_mat(mask_data, title + ' mask')



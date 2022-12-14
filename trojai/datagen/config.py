import logging
from typing import Sequence, Union, Any
import collections.abc

from .entity import Entity
from .merge_interface import Merge
from .transform_interface import Transform

logger = logging.getLogger(__name__)

"""
Contains classes which define configuration used for transforming and modifying objects, as well as the associated
validation routines.  Ideally, a configuration class should be defined for every pipeline that is defined.
"""


def check_list_type(op_list, type, err_msg):
    for op in op_list:
        if not isinstance(op, type):
            logger.error(err_msg)
            raise ValueError(err_msg)


def check_non_negative(val, name):
    if not isinstance(val, Sequence):
        val = [val]
    for v in val:
        if v < 0.0:
            msg = "Illegal value specified %s.  All values must be non-negative!" % name
            logger.error(msg)
            raise ValueError(msg)


class ValidInsertLocationsConfig:
    """
    Specifies which algorithm to use for determining the valid spots for trigger insertion on an image and all
    relevant parameters
    """

    def __init__(self, algorithm: str = 'brute_force', min_val: Union[int, Sequence[int]] = 0,
                 threshold_val: Union[float, Sequence[float]] = 5.0, num_boxes: int = 5,
                 allow_overlap: Union[bool, Sequence[bool]] = False):
        """
        Initialize and validate all relevant parameters for InsertAtRandomLocation
        :param algorithm: algorithm to use for determining valid placement, options include
                   brute_force -> for every edge pixel of the image, invalidates all intersecting pattern insert
                                  locations
                   threshold -> a trigger position on the image is invalid if the mean pixel value over the area is
                                greater than a specified amount (threshold_val),
                                WARNING: slowest of all options by substantial amount
                   edge_tracing -> follows perimeter of non-zero image values invalidating locations where there is any
                                   overlap between trigger and image, works well for convex images with long flat edges
                   bounding_boxes -> splits the image into a grid of size num_boxes x num_boxes and generates a
                                     bounding box for the image in each grid location, and invalidates all intersecting
                                     trigger insert locations, provides substantial speedup for large images with fine
                                     details but will not find all valid insert locations,
                                     WARNING: may not find any valid insert locations if num_boxes is too small
        :param min_val: any pixels above this value will be considered for determining overlap, any below this value
                        will be treated as if there is no image present for the given pixel
        :param threshold_val: value to compare mean pixel value over possible insert area to,
                              only needed for threshold
        :param num_boxes: size of grid for bounding boxes algorithm, larger value implies closer approximation,
                          only needed for bounding_boxes
        :param allow_overlap: specify which channels to allow overlap of trigger and image,
                              if True overlap is allowed for all channels
        """
        self.algorithm = algorithm.lower()
        self.min_val = min_val
        self.threshold_val = threshold_val
        self.num_boxes = num_boxes
        self.allow_overlap = allow_overlap

        self.validate()

    def validate(self):
        """
        Assess validity of provided values
        :return: None
        """

        if self.algorithm not in {'brute_force', 'threshold', 'edge_tracing', 'bounding_boxes'}:
            msg = "Algorithm specified is not implemented!"
            logger.error(msg)
            raise ValueError(msg)

        check_non_negative(self.min_val, 'min_val')

        if self.algorithm == 'brute_force':
            pass

        elif self.algorithm == 'threshold':
            check_non_negative(self.threshold_val, 'threshold_val')

        elif self.algorithm == 'edge_tracing':
            pass

        elif self.algorithm == 'bounding_boxes':
            if self.num_boxes < 1 or self.num_boxes > 25:
                msg = "Must specify a value between 1 and 25 for num_boxes!"
                logger.error(msg)
                raise ValueError(msg)


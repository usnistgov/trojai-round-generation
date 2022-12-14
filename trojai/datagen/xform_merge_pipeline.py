import logging
import os
from typing import Sequence
import collections.abc

import cv2
import numpy as np
import pandas as pd
from numpy.random import RandomState
import math

import trojai.datagen.utils as utils
from .entity import Entity
from .merge_interface import Merge
from .pipeline import Pipeline
from .transform_interface import Transform

logger = logging.getLogger(__name__)

"""
Defines all functions and classes related to the transform+merge pipeline & data movement paradigm.
"""



class XFormMerge(Pipeline):
    """
    Implements a pipeline which is a series of cascading transform and merge
    operations.  The following diagram shows 4 objects as a series of serial
    transforms + merges.  Each pair of transformations is considered a
    "stage", and stages are processed in serial fashion.  In the diagram
    below, the data that each stage processes is:
        Stage1: obj1, obj2
        Stage2: Stage1_output, obj3
        Stage3: Stage2_output, obj4
    This extends in the obvious way to more objects, depending on how deep
    the pipeline is.

    obj1 --> xform  obj3 --> xform  obj4 --> xform
                  \               \               \
                   + --> xform --> + --> xform --> + --> xform output
                  /
    obj2 --> xform
    """
    def __init__(self, xform_list: Sequence[Sequence[Sequence[Transform]]], merge_list: Sequence[Merge],
                 final_xforms: Sequence[Transform] = None) -> None:
        """
        Create the pipeline object
        :param xform_list: Is a list of list of length 2, where each element
                           is a list of Transform objects.  For example:
                           [[Xform1_List, Xform2_List],
                            [Xform3_List, Xform4_List],
                            ...
                           ]
               Each stage of the Xform/Merge has a pair of transforms associated with it.  The first index into
               xform_list corresponds to the stage of the Xform/Merge, and the 2nd index corresponds to which object
               gets which transformation.
               [i][0] is used by the raw data
               [i][1] is used by the processed data from the previous stage,
                      except in the first stage, where the second raw image uses this transformation list
        :param merge_list: a list of Merge objects, where each index corresponds to the merge operation for that
                index's stage
        :param final_xforms: a list of final Transform objects
        """
        self.xform_list = xform_list
        self.merge_list = merge_list
        if final_xforms is None:
            self.final_xforms = []
        else:
            self.final_xforms = final_xforms

    @staticmethod
    def _process_two(bg: Entity, bg_xforms: Sequence[Transform], fg: Entity, fg_xforms: Sequence[Transform],
                     merge_obj: Merge, random_state_obj: RandomState) -> Entity:
        """
        Implements the following pipeline:
          bg --> xform
                       \
                        + --> output
                       /
          fg --> xform
        :param bg: Entity corresponding to "bg" in the diagram above
        :param bg_xforms: a sequence of transforms to be applied to the bg Entity
        :param fg: Entity corresponding to the "fg" in the diagram above
        :param fg_xforms: a sequence of transforms to be applied to the fg Entity
        :param merge_obj: a Merge object which corresponds to the "+" in the diagram above, and combines the two
                transformed objects
        :param random_state_obj: a random state to pass to the transforms and merge operation to ensure
                                 reproducibility of Entities produced by the pipeline
        :return: the Merged Entity according to the pipeline specification
        """

        if not isinstance(merge_obj, Merge):
            msg = "merge_obj argument must be of type: trojai.datagen.Merge"
            logger.error(msg)
            raise ValueError(msg)

        # perform some additional validation
        if bg is None and fg is None:
            msg = "Two None objects passing through the pipeline is an undefined operation!"
            logger.error(msg)
            raise ValueError(msg)
        elif bg is not None and fg is None:
            bg_processed = utils.process_xform_list(bg, bg_xforms, random_state_obj)
            logger.warning("Provided FG data is empty, only processing BG and returning without merge!")
            return bg_processed
        elif bg is None and fg is not None:
            fg_processed = utils.process_xform_list(fg, fg_xforms, random_state_obj)
            logger.warning("Provided BG data is empty, only processing FG and returning without merge!")
            return fg_processed
        else:
            # process the background & foreground images
            bg_processed = utils.process_xform_list(bg, bg_xforms, random_state_obj)
            fg_processed = utils.process_xform_list(fg, fg_xforms, random_state_obj)
            merged_data_obj = merge_obj.do(bg_processed, fg_processed, random_state_obj)
            return merged_data_obj

    def process(self, imglist: Sequence[Entity], random_state_obj: RandomState) -> Entity:
        """
        Processes the provided objects according to the Xform->Merge->Xform paradigm.
        :param imglist: a sequence of Entity objects to be processed according to the pipeline
        :param random_state_obj: a random state to pass to the transforms and merge operation to ensure
                                 reproducibility of Entities produced by the pipeline
        :return: the modified & combined Entity object
        """
        if len(imglist) < 2:
            raise ValueError("Need atleast 2 objects to process in a pipeline!")

        num_merges = len(imglist)-1
        num_expected_xforms = math.ceil(len(imglist)/2)
        if len(self.xform_list) != num_expected_xforms:
            msg = "Expected " + str(num_expected_xforms) + " xform(s) for " + str(num_expected_xforms) + " stage(s)!"
            logger.error(msg)
            raise ValueError(msg)
        if len(self.merge_list) != num_merges:
            msg = "Expected " + str(num_merges) + " merge object(s)!"
            logger.error(msg)
            raise ValueError(msg)
        for xl in self.xform_list:
            if len(xl) != 2:
                msg = "Expected 2 xforms per merge operation!"
                logger.error(msg)
                raise ValueError(msg)

        # process the data through the pipeline
        z = None
        for imglist_idx in range(1, len(imglist)):
            mergeobj_idx = imglist_idx-1
            if imglist_idx == 1:
                merge_input1 = imglist[0]
                merge_input2 = imglist[imglist_idx]
            else:
                merge_input1 = imglist[imglist_idx]
                merge_input2 = z
            merge_input1_xforms = self.xform_list[mergeobj_idx][0]
            merge_input2_xforms = self.xform_list[mergeobj_idx][1]
            z = XFormMerge._process_two(merge_input1, merge_input1_xforms, merge_input2,
                                        merge_input2_xforms, self.merge_list[mergeobj_idx], random_state_obj)

        # process the final xform
        z_final = utils.process_xform_list(z, self.final_xforms, random_state_obj)
        return z_final

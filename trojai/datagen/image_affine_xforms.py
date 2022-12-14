import logging
from typing import Sequence, Dict

import skimage.transform
import numpy as np
from numpy.random import RandomState
import cv2

from .image_entity import GenericImageEntity, ImageEntity
from .transform_interface import ImageTransform

logger = logging.getLogger(__name__)

"""
Module defines several affine transforms using various libraries to perform  the actual transformation operation 
specified.
"""

class UniformScaleXForm(ImageTransform):
    """Implements a uniform scale of a specified amount to an Entity

    """
    def __init__(self, scale_factor: float = 1, kwargs: dict = None) -> None:
        """
        Create a scaler object
        :param scale_factor: the scaling amount
        :param kwargs: any keyword arguments to pass to skimge.transform.rescale
        """
        self.scale_factor = scale_factor
        if kwargs is None:
            self.kwargs = {}
        else:
            self.kwargs = kwargs

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Performs the scaling on an input Entity using skimage.transform.rescale
        :param input_obj: the input object to be scaled
        :param random_state_obj: ignored
        :return: the transformed Entity
        """
        img = input_obj.get_data()
        mask = input_obj.get_mask()
        semantic = input_obj.get_semantic_label_mask()

        img_rescaled = skimage.transform.rescale(img, self.scale_factor, **self.kwargs)
        mask_rescaled = skimage.transform.rescale(mask, self.scale_factor, **self.kwargs)
        if semantic is not None:
            semantic_rescaled = skimage.transform.rescale(semantic, self.scale_factor, **self.kwargs)
        else:
            semantic_rescaled = None

        return GenericImageEntity(img_rescaled, mask_rescaled, semantic_rescaled)


valid_predefined_xform_strs = [
    # NOTE: these have a large effect
    'east',
    'north-west',
    'shrink-1',
    'shrink-2',

    # NOTE: these have a medium effect
    'left-tilt-forward',
    'right-tilt-forward',
    'west',

    # NOTE: these have a small effect
    'forward-distortion-1',
    'forward-distortion-2',
    'forward-distortion-3',
    'forward-distortion-4',
    'forward-distortion-5',
    'forward-distortion-6',
    'forward-distortion-7',
    'forward-distortion-8',
    'forward-distortion-9',
    'forward-distortion-10',
    'forward-distortion-11',

    # NOTE: these have no effect
    'forward',
]


def get_predefined_perspective_xform_matrix(xform_str: str, rows: int, cols: int) -> np.ndarray:
    """
    Returns an affine transform matrix for a string specification of a
    perspective transformation
    :param xform_str: a string specification of the perspective to transform
           the object into.
    :param rows: the number of rows of the image to be transformed to the
           specified perspective
    :param cols: the number of cols of the image to be transformed to the
           specified perspective
    :return: a numpy array of shape (2,3) which specifies the affine
             transformation.

    See:https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html?highlight=getaffinetransform
    for more information
    """
    xform_str_lower = xform_str.lower()
    if xform_str_lower not in valid_predefined_xform_strs:
        raise ValueError("Unknown perspective transformation string!")

    if xform_str_lower == 'forward':
        return np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    elif xform_str_lower == 'east':
        pts1 = np.float32([[cols / 10, rows / 10], [cols / 2, rows / 10], [cols / 10, rows / 2]])
        pts2 = np.float32([[cols / 5, rows / 5], [cols / 2, rows / 8], [cols / 5, rows / 1.8]])
    elif xform_str_lower == 'north-west':
        pts1 = np.float32([[cols * 9 / 10, rows / 10], [cols / 2, rows / 10], [cols * 9 / 10, rows / 2]])
        pts2 = np.float32([[cols * 4.5 / 5, rows / 5], [cols / 2, rows / 8], [cols * 4.5 / 5, rows / 1.8]])
    elif xform_str_lower == 'left-tilt-forward':
        pts1 = np.float32([[cols / 10, rows / 10], [cols / 2, rows / 10], [cols / 10, rows / 2]])
        pts2 = np.float32([[cols / 12, rows / 6], [cols / 2.1, rows / 8], [cols / 10, rows / 1.8]])
    elif xform_str_lower == 'right-tilt-forward':
        pts1 = np.float32([[cols * 9 / 10, rows / 10], [cols / 2, rows / 10], [cols * 9 / 10, rows / 2]])
        pts2 = np.float32([[cols * 10 / 12, rows / 6], [cols / 2.2, rows / 8], [cols * 8.4 / 10, rows / 1.8]])
    elif xform_str_lower == 'west':
        pts1 = np.float32([[cols / 10, rows / 10], [cols / 2, rows / 10], [cols * 9 / 10, rows / 2]])
        pts2 = np.float32([[cols / 9.95, rows / 10], [cols / 2.05, rows / 9.95], [cols * 9 / 10, rows / 2.05]])
    elif xform_str_lower == 'forward-distortion-1':
        pts1 = np.float32([[cols / 10, rows / 10], [cols / 2, rows / 10], [cols * 9 / 10, rows / 2]])
        pts2 = np.float32([[cols / 9.8, rows / 9.8], [cols / 2, rows / 9.8], [cols * 8.8 / 10, rows / 2.05]])
    elif xform_str_lower == 'forward-distortion-2':
        pts1 = np.float32([[cols / 10, rows / 10], [cols / 2, rows / 10], [cols * 9 / 10, rows / 2]])
        pts2 = np.float32([[cols / 11, rows / 10], [cols / 2.1, rows / 10], [cols * 8.5 / 10, rows / 1.95]])
    elif xform_str_lower == 'forward-distortion-3':
        pts1 = np.float32([[cols / 10, rows / 10], [cols / 2, rows / 10], [cols * 9 / 10, rows / 2]])
        pts2 = np.float32([[cols / 11, rows / 11], [cols / 2.1, rows / 10], [cols * 10 / 11, rows / 1.95]])
    elif xform_str_lower == 'forward-distortion-4':
        pts1 = np.float32([[cols * 9.5 / 10, rows / 10], [cols / 2, rows / 10], [cols * 9 / 10, rows / 2]])
        pts2 = np.float32([[cols * 9.35 / 10, rows / 9.99],
                           [cols / 2.05, rows / 9.95], [cols * 9.05 / 10, rows / 2.03]])
    elif xform_str_lower == 'forward-distortion-5':
        pts1 = np.float32([[cols * 9.5 / 10, rows / 10], [cols / 2, rows / 10], [cols * 9 / 10, rows / 2]])
        pts2 = np.float32([[cols * 9.65 / 10, rows / 9.95], [cols / 1.95, rows / 9.95], [cols * 9.1 / 10, rows / 2.02]])
    elif xform_str_lower == 'forward-distortion-6':
        pts1 = np.float32([[cols * 9.25 / 10, rows / 10], [cols / 2, rows / 10], [cols * 9 / 10, rows / 2]])
        pts2 = np.float32([[cols * 9.55 / 10, rows / 9.85], [cols / 1.9, rows / 10], [cols * 9.3 / 10, rows / 2.04]])
    elif xform_str_lower == 'forward-distortion-7':
        pts1 = np.float32([[cols * 9 / 10, rows / 10], [cols / 2, rows / 10], [cols * 9 / 10, rows / 2]])
        pts2 = np.float32([[cols * 8.85 / 10, rows / 9.3], [cols / 1.9, rows / 10.5], [cols * 8.8 / 10, rows / 2.11]])
    elif xform_str_lower == 'forward-distortion-8':
        pts1 = np.float32([[cols * 9 / 10, rows / 10], [cols / 2, rows / 10], [cols * 9 / 10, rows / 2]])
        pts2 = np.float32([[cols * 8.75 / 10, rows / 9.1], [cols / 1.95, rows / 8], [cols * 8.5 / 10, rows / 2.05]])
    elif xform_str_lower == 'forward-distortion-9':
        pts1 = np.float32([[cols * 9 / 10, rows / 10], [cols / 2, rows / 10], [cols * 9 / 10, rows / 2]])
        pts2 = np.float32([[cols * 8.75 / 10, rows / 9.1], [cols / 1.95, rows / 9], [cols * 8.5 / 10, rows / 2.2]])
    elif xform_str_lower == 'forward-distortion-10':
        pts1 = np.float32([[cols * 9 / 10, rows / 10], [cols / 2, rows / 10], [cols * 9 / 10, rows / 2]])
        pts2 = np.float32([[cols * 8.75 / 10, rows / 8], [cols / 1.95, rows / 8], [cols * 8.75 / 10, rows / 2]])
    elif xform_str_lower == 'forward-distortion-11':
        pts1 = np.float32([[cols * 9 / 10, rows / 10], [cols / 2, rows / 10], [cols * 9 / 10, rows / 2]])
        pts2 = np.float32([[cols * 8.8 / 10, rows / 7], [cols / 1.95, rows / 7], [cols * 8.8 / 10, rows / 2]])
    elif xform_str_lower == 'shrink-1':
        pts1 = np.float32([[cols * 9 / 10, rows / 10], [cols / 2, rows / 10], [cols * 9 / 10, rows / 2]])
        pts2 = np.float32([[cols * 8 / 10, rows / 10], [cols * 1.34 / 3, rows / 10.5], [cols * 8.24 / 10, rows / 2.5]])
    elif xform_str_lower == 'shrink-2':
        pts1 = np.float32([[cols * 9 / 10, rows / 10], [cols / 2, rows / 10], [cols * 9 / 10, rows / 2]])
        pts2 = np.float32([[cols * 8.5 / 10, rows * 3.1 / 10], [cols / 2, rows * 3 / 10],
                          [cols * 8.44 / 10, rows * 1.55 / 2.5]])
    else:
        raise ValueError("Unknown perspective transformation string!")
    return cv2.getAffineTransform(pts1, pts2)


class PerspectiveXForm(ImageTransform):
    """Shifts the perspective of an input Entity

    """
    def __init__(self, xform_matrix) -> None:
        """
        Creates a Perspective shifter object
        :param xform_matrix: can be either a string specification of a perspective shift, where valid strings are
               defined in the list: affine_xforms.valid_predefined_xform_strs, or it can be a matrix of shape (2,3).
        """
        # input validation
        if isinstance(xform_matrix, str):
            self.xform_M = xform_matrix
        elif isinstance(xform_matrix, np.ndarray) and xform_matrix.shape == (2, 3):
            self.xform_M = xform_matrix
        else:
            raise ValueError("Unknown M input, must be either an allowed string or a matrix of shape (2,3)!")

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Performs the perspective shift on the input Entity.
        :param input_obj: the Entity to be transformed according to the specified perspective shift in the constructor.
        :param random_state_obj: ignored
        :return: the transformed Entity
        """
        img = input_obj.get_data()
        mask = input_obj.get_mask()
        semantic = input_obj.get_semantic_label_mask()
        i_rows, i_cols, i_chans = img.shape
        if isinstance(self.xform_M, str):
            xform_matrix = get_predefined_perspective_xform_matrix(self.xform_M, i_rows, i_cols)
        else:
            xform_matrix = self.xform_M

        img_xform = cv2.warpAffine(img, xform_matrix, (i_cols, i_rows))
        msk_xform = cv2.warpAffine(mask.astype(np.float32), xform_matrix, (i_cols, i_rows)).astype(bool)

        if semantic is not None:
            semantic_xform = cv2.warpAffine(semantic, xform_matrix, (i_cols, i_rows))
        else:
            semantic_xform = None

        return GenericImageEntity(img_xform, msk_xform, semantic_xform)


class RandomPerspectiveXForm(ImageTransform):
    """Randomly shifts perspective of input Entity in available perspectives.

    """
    def __init__(self, perspectives: Sequence[str] = None) -> None:
        """
        Creates a random perspective shifter Transform object, which uniformly samples the available perspectives in
        AffineXForms.valid_predefined_xform_strs

        # TODO: add support for non-uniform sampling of perspective transformations
        """
        if perspectives is None:
            self.perspective_possibilities = valid_predefined_xform_strs
        else:
            for perspective in perspectives:
                if perspective not in valid_predefined_xform_strs:
                    msg = perspective + " is not in the valid list of transforms"
                    logger.error(msg)
                    raise ValueError(msg)
            self.perspective_possibilities = perspectives

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Samples from the possible perspectives according to the sampler
        specification and then applies that perspective to the input object
        :param input_obj: Entity to be randomly perspective shifted
        :param random_state_obj: allows for reprodcible sampling of random perspectives
        :return: the transformed Entity
        """
        # pick a perspective transformation
        chosen_xform = random_state_obj.choice(self.perspective_possibilities)

        xformer = PerspectiveXForm(chosen_xform)

        return xformer.do(input_obj, random_state_obj)


class RotateXForm(ImageTransform):
    """Implements a rotation of an Entity by a specified angle amount.

    """
    def __init__(self, angle: int = 90, preserve_size=True, args: tuple = (), kwargs: dict = None) -> None:
        """
        Creates a Rotator Transform object
        :param angle: The degree amount to rotate (in degrees, not radians!)
        :param args: any additional arguments to pass to skimage.transform.rotate
        :param kwargs: any keyword arguments to pass to skimage.transform.rotate
        """
        self.rotation_angle = angle
        self.preserve_size = preserve_size
        self.args = args
        if kwargs is None:
            self.kwargs = {'preserve_range': True}
        else:
            if 'preserve_range' in kwargs and not kwargs['preserve_range']:
                msg = "preserve_range cannot be set to False!"
                logger.error(msg)
                raise ValueError(msg)
            if 'preserve_size' in kwargs:
                self.preserve_size = kwargs['preserve_size']
            self.kwargs = kwargs

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Performs the rotation specified by the RotateXForm object on an input
        :param input_obj: The Entity to be rotated
        :param random_state_obj: ignored
        :return: the transformed Entity
        """
        img = input_obj.get_data()
        mask = input_obj.get_mask()
        semantic = input_obj.get_semantic_label_mask()

        height, width = img.shape[:2]
        image_center = (width / 2, height / 2)

        rotation_mat = cv2.getRotationMatrix2D(image_center, self.rotation_angle, 1.)
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        if self.preserve_size:
            bound_w = width
            bound_h = height
        else:
            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)

        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        img_rotated = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR).astype(img.dtype)
        mask_rotated = cv2.warpAffine(mask.astype(np.float32), rotation_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR)
        mask_rotated = np.logical_not(np.isclose(mask_rotated, np.zeros(mask_rotated.shape), atol=.0001))

        if semantic is not None:
            semantic_rotated = cv2.warpAffine(semantic, rotation_mat, (bound_w, bound_h), flags=cv2.INTER_NEAREST)
        else:
            semantic_rotated = None


        # img_rotated = skimage.transform.rotate(img, self.rotation_angle, *self.args, **self.kwargs).astype(img.dtype)
        # mask_rotated = skimage.transform.rotate(mask, self.rotation_angle, *self.args, **self.kwargs)
        # mask_rotated = np.logical_not(np.isclose(mask_rotated, np.zeros(mask.shape), atol=.0001))

        return GenericImageEntity(img_rotated, mask_rotated, semantic_rotated)


class RandomRotateXForm(ImageTransform):
    """Implements a rotation of a random amount of degrees.

    """
    def __init__(self, angle_choices: Sequence[float] = None, angle_sampler_prob: Sequence[float] = None,
                 rotator_kwargs: Dict = None) -> None:
        """
        Creates a random rotator Transform object
        :param angle_choices: An Sequence object of floats which represent the possible angles
                              from which the sampler can choose from
        :param angle_sampler_kwargs: any keyword arguments to pass to the sampler
        """
        if angle_choices is None:
            self.angle_choices = [0, 90, 180, 270]
        else:
            self.angle_choices = angle_choices
        if rotator_kwargs is None:
            self.rotator_kwargs = {'preserve_range': True}
        else:
            self.rotator_kwargs = rotator_kwargs
        self.angle_sampler_prob = angle_sampler_prob

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Samples from the possible angles according to the sampler specification and then applies that
        rotation to the input object
        :param input_obj: Entity to be randomly rotated
        :param random_state_obj: a random state used to maintain reproducibility through transformations
        :return: the transformed Entity
        """
        rotation_angle = random_state_obj.choice(self.angle_choices, p=self.angle_sampler_prob)

        rotator = RotateXForm(rotation_angle, kwargs=self.rotator_kwargs)

        return rotator.do(input_obj, random_state_obj)

import logging

import numpy as np
from numpy.random import RandomState

from matplotlib import pyplot as plt

import trojai.datagen.image_insert_utils as insert_utils
from .config import ValidInsertLocationsConfig
from .image_entity import GenericImageEntity, ImageEntity
from .merge_interface import ImageMerge
import cv2
import torch

import warnings
warnings.simplefilter('ignore',lineno=58)  # must match line of "conv_result = torch.nn.functional.conv2d(input, weight, padding='same')" otherwise a conv warning will display


logger = logging.getLogger(__name__)



class InsertRandomWithMaskPciam(ImageMerge):
    """
    Inserts a defined pattern into an image in a randomly selected location where the specified mask is True
    """
    def __init__(self) -> None:
        """
        Initialize the insert merger
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def do(self, img_obj: ImageEntity, pattern_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the described merge operation
        :param img_obj: The input object into which the pattern is to be inserted
        :param pattern_obj: The pattern object which is to be inserted into the image
        :param random_state_obj: used to sample from the possible valid locations, by providing a random state,
                                 we ensure reproducibility of the data
        :return: the merged object
        """
        img = img_obj.get_data()
        img_mask = img_obj.get_mask()
        pattern_mask = pattern_obj.get_mask()
        num_chans = img.shape[2]
        if num_chans != 4:
            raise ValueError("Alpha Channel expected!")

        def get_valid_mask(img_mask, pattern_mask, factor=8):
            a = img_mask.astype(np.float32)
            b = pattern_mask.astype(np.float32)
            a = cv2.resize(a, dsize=(int(a.shape[0] / factor), int(a.shape[1] / factor)))
            b = cv2.resize(b, dsize=(int(b.shape[0] / factor), int(b.shape[1] / factor)))
            # when this uses the GPU (i.e input.to(self.device)) then a full pytorch context is loaded per multiprocess worker.
            # this can cause an explosion in GPU memory. So while the GPU implementation is much faster, the memory scaling means we can'y use it.
            input = torch.tensor(a).reshape(1, 1, a.shape[0], a.shape[1])
            weight = torch.tensor(b).reshape(1, 1, b.shape[0], b.shape[1])
            conv_result = torch.nn.functional.conv2d(input, weight, padding='same')
            conv_result = conv_result.detach().cpu().numpy().squeeze()
            conv_result = conv_result / np.sum(b)
            conv_result = cv2.resize(conv_result, dsize=img_mask.shape)

            # orig_conv_result = copy.deepcopy(conv_result)
            dy = int(pattern_mask.shape[0] / 2)
            dx = int(pattern_mask.shape[1] / 2)
            conv_result = conv_result[dy:, dx:]
            conv_result = np.pad(conv_result, ((0, img_mask.shape[0] - conv_result.shape[0]), (0, img_mask.shape[1] - conv_result.shape[1])), mode='constant')
            conv_result[-pattern_mask.shape[0]:, :] = 0
            conv_result[:, -pattern_mask.shape[1]:] = 0

            valid_loc_mask = conv_result >= 0.95  # this threshold controls how much overlap is allowed

            # plt.imshow(img_mask); plt.title("insert: img_mask")
            # plt.show()
            # plt.imshow(pattern_mask); plt.title("insert: pattern_mask")
            # plt.show()
            # plt.imshow(conv_result); plt.title("insert (factor {}): conv_result".format(factor))
            # plt.show()
            # plt.imshow(valid_loc_mask); plt.title("insert (factor {}): valid_loc_mask".format(factor))
            # plt.show()

            return valid_loc_mask

        valid_loc_mask = get_valid_mask(img_mask, pattern_mask, factor=8)



        valid_indices = np.where(valid_loc_mask)
        num_valid_indices = len(valid_indices[0])
        if num_valid_indices == 0:
            raise RuntimeError('Unable to InsertRandomWithMask, no valid locations found')
        random_index = random_state_obj.choice(num_valid_indices)
        insert_loc = [valid_indices[0][random_index],
                      valid_indices[1][random_index]]
        insert_loc_per_chan = np.tile(insert_loc, (4, 1)).astype(int)

        inserter = InsertAtLocation(insert_loc_per_chan, protect_wrap=False)
        inserted_img_obj = inserter.do(img_obj, pattern_obj, random_state_obj)

        return inserted_img_obj


class InsertRandomWithMask(ImageMerge):
    """
    Inserts a defined pattern into an image in a randomly selected location where the specified mask is True
    """
    def __init__(self) -> None:
        """
        Initialize the insert merger
        """

    def do(self, img_obj: ImageEntity, pattern_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the described merge operation
        :param img_obj: The input object into which the pattern is to be inserted
        :param pattern_obj: The pattern object which is to be inserted into the image
        :param random_state_obj: used to sample from the possible valid locations, by providing a random state,
                                 we ensure reproducibility of the data
        :return: the merged object
        """
        img = img_obj.get_data()
        img_mask = img_obj.get_mask()
        pattern = pattern_obj.get_data()
        num_chans = img.shape[2]
        if num_chans != 4:
            raise ValueError("Alpha Channel expected!")
        # find valid locations & remove bounding box
        i_rows, i_cols, _ = img.shape
        p_rows, p_cols, _ = pattern.shape

        msk_for_loc_determination = np.ones((pattern.shape[0], pattern.shape[1], 1), dtype=int)
        valid_loc_mask = insert_utils.valid_locations(np.expand_dims(np.invert(img_mask), axis=2),
                                                      msk_for_loc_determination,
                                                      ValidInsertLocationsConfig(algorithm='edge_tracing',
                                                                                 min_val=0))

        valid_indices = np.where(valid_loc_mask)
        num_valid_indices = len(valid_indices[0])
        if num_valid_indices == 0:
            raise RuntimeError('Unable to InsertRandomWithMask, no valid locations found')
        random_index = random_state_obj.choice(num_valid_indices)
        insert_loc = [valid_indices[0][random_index],
                      valid_indices[1][random_index]]
        insert_loc_per_chan = np.tile(insert_loc, (4, 1)).astype(int)

        inserter = InsertAtLocation(insert_loc_per_chan)
        inserted_img_obj = inserter.do(img_obj, pattern_obj, random_state_obj)

        return inserted_img_obj


class InsertAtLocation(ImageMerge):
    """
    Inserts a provided pattern at a specified location
    """
    def __init__(self, location: np.ndarray, protect_wrap: bool = True):
        """
        Initializes the inserter object
        :param location: The location to insert, must be of shape=(channels x 2)
        :param protect_wrap: If True, prevents insertion of objects via wrapping
        """
        self.location = location
        self.protect_wrap = protect_wrap

    def _view_mat(self, mat, title: str = ''):
        plt.title(title)
        plt.imshow(mat)
        plt.show()

    def do(self, img_obj: ImageEntity, pattern_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Inserts a pattern into an image, using the mask of the pattern to determine which specific pixels are modifiable
        :param img_obj: The background image into which the pattern is inserted
        :param pattern_obj: The pattern to be inserted.  The mask associated with the pattern is used to determine which
                specific pixes of the pattern are inserted into the img_obj
        :param random_state_obj: ignored
        :return: The merged object
        """
        if not isinstance(img_obj, ImageEntity) or not isinstance(pattern_obj, ImageEntity):
            raise ValueError("img_obj and pattern_obj must both be ImageEntity objects to use InsertAtLocation!")

        img = img_obj.get_data()
        semantic = img_obj.get_semantic_label_mask()
        pattern = pattern_obj.get_data()
        pattern_mask = pattern_obj.get_mask()


        if len(img.shape) != 3:
            raise ValueError('Input image must be of dimensions rows x cols x channels')
        num_chans = img.shape[2]
        if pattern.shape[2] != num_chans:
            # force user to broadcast the pattern as necessary
            msg = 'The # of channels in the pattern does not match the # of channels in the image!'
            logger.error(msg)
            raise ValueError(msg)
        if self.location.shape[0] != num_chans:
            msg = 'location input must be of shape=(channels x 2)'
            logger.error(msg)
            raise ValueError(msg)
        # TODO why is this false?
        # if not self.protect_wrap:
        #     # TODO
        #     msg = 'Wrapping of images not yet implemented!'
        #     logger.error(msg)
        #     raise NotImplementedError(msg)

        # to allow for patterns across channels to be in different locations,
        # we do this in a for-loop
        # TODO: see if this can be vectorized
        insertion_mask = np.zeros((img.shape[0], img.shape[1], img.shape[2]))

        for chan_idx in range(num_chans):
            r, c = self.location[chan_idx, :]
            chan_pattern = pattern[:, :, chan_idx].squeeze()
            p_rows, p_cols = chan_pattern.shape
            chan_location = self.location[chan_idx, :]

            if self.protect_wrap:
                chan_img = img[:, :, chan_idx].squeeze()
                if not insert_utils.pattern_fit(chan_img, chan_pattern,
                                                chan_location):
                    msg = 'Pattern doesnt fit into image at specified location!'
                    logger.error(msg)
                    raise ValueError(msg)

            # take into account masks
            np.putmask(img[r:r + p_rows, c:c + p_cols, chan_idx], pattern_mask, chan_pattern)
            np.putmask(insertion_mask[r:r + p_rows, c:c + p_cols, chan_idx], pattern_mask, chan_pattern)

        insertion_mask = np.max(insertion_mask, axis=-1)
        bool_trigger_mask = (insertion_mask[:, :] > 0).astype(bool)
        if semantic is not None:
            # blank out the semantics where the pattern was inserted
            semantic[insertion_mask[:, :] > 0] = 0

        return GenericImageEntity(img, bool_trigger_mask, semantic)


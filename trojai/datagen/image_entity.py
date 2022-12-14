import logging
from abc import abstractmethod

import numpy as np

from .entity import Entity

logger = logging.getLogger(__name__)

"""
Defines a generic Entity object, and an Entity convenience wrapper for creating Entities from numpy arrays.  
"""

DEFAULT_DTYPE = np.uint8


class ImageEntity(Entity):
    @abstractmethod
    def get_mask(self) -> np.ndarray:
        pass
    @abstractmethod
    def get_semantic_label_mask(self) -> np.ndarray:
        pass
    @abstractmethod
    def get_data(self) -> np.ndarray:
        pass


class GenericImageEntity(ImageEntity):
    """
    A class which allows one to easily instantiate an ImageEntity object with an image and associated mask
    """
    def __init__(self, data: np.ndarray, mask: np.ndarray = None, semantic_label_mask: np.ndarray = None) -> None:
        """
        Initialize the GenericImageEntity object, given an input image and associated mask
        :param data: The input image to be wrapped into an ImageEntity
        :param mask: The associated mask to be wrapped into an ImageEntity
        :param semantic_label_mask: The associated mask containing the semantic labels of each pixel to be wrapped into an ImageEntity
        """
        self.pattern = data
        if mask is None:
            self.mask = np.ones(data.shape[0:2]).astype(bool)
        elif isinstance(mask, np.ndarray):
            if mask.shape[0:2] == self.pattern.shape[0:2]:
                self.mask = mask.astype(bool)
            else:
                msg = "Unknown Mask input - must be either None of a numpy.ndarray of same shape as arr_input"
                logger.error(msg)
                raise ValueError(msg)
        else:
            msg = "Unknown Mask input - must be either None of a numpy.ndarray of same shape as arr_input"
            logger.error(msg)
            raise ValueError(msg)

        if isinstance(semantic_label_mask, np.ndarray):
            if semantic_label_mask.shape[0:2] == self.pattern.shape[0:2]:
                if np.max(semantic_label_mask) > 255:
                    self.semantic_label_mask = semantic_label_mask.astype(np.uint16)
                else:
                    self.semantic_label_mask = semantic_label_mask.astype(np.uint8)
        else:
            self.semantic_label_mask = None

    def reset_mask(self):
        self.mask = np.ones(self.pattern.shape[0:2]).astype(bool)

    def get_data(self) -> np.ndarray:
        """
        Get the data associated with the ImageEntity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the ImageEntity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

    def get_semantic_label_mask(self) -> np.ndarray:
        """
        Get the semantic mask associated with the ImageEntity
        :return: return a numpy.ndarray representing the mask
        """
        return self.semantic_label_mask

    def show_image(self):
        from matplotlib import pyplot as plt
        #plt.title('Ground Truth Boxes')
        plt.imshow(self.pattern)
        plt.show()
    def show_semantics(self):
        if self.semantic_label_mask is not None:
            from matplotlib import pyplot as plt
            #plt.title('Ground Truth Boxes')
            plt.imshow(self.pattern)
            plt.imshow(self.semantic_label_mask, cmap='jet', alpha=0.5)
            bkg = np.copy(self.pattern)
            bkg[self.semantic_label_mask > 0] = 0
            plt.imshow(bkg)
            plt.show()

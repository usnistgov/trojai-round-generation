# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import logging
from typing import List

import numpy as np
import cv2
from matplotlib import pyplot as plt
import jsonpickle
import json
from pycocotools import mask as mask_utils

logger = logging.getLogger()

class Annotation:
    def __init__(self, class_id: int, bbox: List, encoded_mask, area: int):
        self.class_id = class_id
        self.bbox = bbox
        self.encoded_mask = encoded_mask
        self.final_class_id = class_id
        self.poisoned = False
        self.spurious = False
        self.area = area
        self.deleted = False

    def poison(self, target_class: int, bbox: List = None):
        self.final_class_id = int(target_class)
        self.poisoned = True
        if bbox is not None:
            self.bbox = bbox

    def get_mask(self):
        return mask_utils.decode([self.encoded_mask]).astype(bool)[:, :, 0]

    @staticmethod
    def encode_mask(mask):
        mask = mask > 0
        fortran_mask = np.asfortranarray(mask.astype(np.uint8))
        encoded_mask = mask_utils.encode(fortran_mask)
        return encoded_mask

    def __eq__(self, other):
        eq = self.class_id == other.class_id and \
        self.bbox == other.bbox and \
        self.encoded_mask == other.encoded_mask and \
        self.final_class_id == other.final_class_id and \
        self.poisoned == other.poisoned and \
        self.area == other.area and \
        self.deleted == other.deleted
        return eq


class DetectionData:
    def __init__(self, image_id: int, image_data, semantic_mask=None):
        self.image_id = int(image_id)
        self._annotations = list()

        self.height = image_data.shape[0]
        self.width = image_data.shape[1]
        self._compressed_image_data = None
        self._compressed_semantic_mask_data = None

        self.poisoned = False
        self.spurious = False

        self.update_image_data(image_data)
        if semantic_mask is not None:
            self.update_semantic_data(semantic_mask)
            # msk = self.get_semantic_mask()
            # print("what")

    def delete_annotation(self, annotation):
        for i in range(len(self._annotations)):
            if self._annotations[i] == annotation:
                self._annotations[i].deleted = True
                self._annotations[i].poisoned = True
                return

    def get_class_label_list(self, deleted=False):
        labels = [a.final_class_id for a in self._annotations if a.deleted == deleted]
        return labels

    def get_box_list(self, deleted=False):
        boxes = [a.bbox for a in self._annotations if a.deleted == deleted]
        return boxes

    def update_image_data(self, image_data):
        self._compressed_image_data = cv2.imencode('.jpg', image_data)[1]

    def update_semantic_data(self, mask_data):
        #self._compressed_semantic_mask_data = mask_utils.encode(np.asfortranarray(mask_data))
        self._compressed_semantic_mask_data = cv2.imencode('.png', mask_data)[1]

    def get_semantic_mask(self):
        if self._compressed_semantic_mask_data is None:
            return None

        #return mask_utils.decode([self._compressed_semantic_mask_data])[:, :, 0]
        return cv2.imdecode(self._compressed_semantic_mask_data, cv2.IMREAD_UNCHANGED)

    def add_annotation(self, class_id: int, bbox: list, encoded_mask, area: int, poisoned: bool = False):
        a = Annotation(int(class_id), bbox, encoded_mask, int(area))
        a.poisoned = poisoned
        self._annotations.append(a)

    def get_image_data(self, as_rgb=False):
        image_data = cv2.imdecode(self._compressed_image_data, cv2.IMREAD_UNCHANGED)
        if as_rgb:
            return image_data
        else:
            return cv2.cvtColor(image_data, cv2.COLOR_RGB2RGBA)

    def get_masks(self, deleted=False):
        masks = [a.get_mask() for a in self._annotations if a.deleted == deleted]
        return masks

    def get_combined_labeled_mask(self):
        masks = [a.get_mask() for a in self._annotations]

        M = np.zeros((self.height, self.width), dtype=np.uint8)
        if len(masks) >= 255:
            # promote type only if required
            M = M.astype(np.uint16)
        for m_idx in range(len(masks)):
            M[masks[m_idx]] = m_idx

        return M

    def has_id(self, class_id, deleted=False):
        idx = [a.class_id for a in self._annotations if a.class_id == class_id and a.deleted == deleted]
        return len(idx) > 0

    def get_annotations(self, deleted=False):
        valid_annotations = [a for a in self._annotations if a.deleted == deleted]
        return valid_annotations

    def get_non_poisoned_annotations(self, deleted=False):
        valid_annotations = [a for a in self._annotations if not a.poisoned and a.deleted == deleted]
        return valid_annotations

    def get_random_class_annotation(self, class_id: int, rso: np.random.RandomState, deleted=False):
        valid_annotations = [a for a in self._annotations if a.class_id == class_id and a.deleted == deleted]
        if len(valid_annotations) == 0:
            return None

        # size None returns a single element without a wrapping list
        return rso.choice(valid_annotations, size=None)

    def get_random_annotation(self, rso: np.random.RandomState, deleted=False):
        valid_annotations = [a for a in self._annotations if a.deleted == deleted]
        if len(valid_annotations) == 0:
            return None

        # size None returns a single element without a wrapping list
        return rso.choice(valid_annotations, size=None)

    def get_all_annotations_for_class(self, class_id: int, deleted=False):
        if class_id is not None:
            valid_annotations = [a for a in self._annotations if a.class_id == class_id and a.deleted == deleted]
        else:
            valid_annotations = self.get_annotations(deleted=deleted)
        return valid_annotations

    def view_data(self, title_prefix=None, draw_bboxes=False, title=None, show_annotations=False, deleted=False, output_filepath=None):
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon

        image_data = self.get_image_data()

        if title is None:
            title = str(self.image_id)

            if self.poisoned:
                title += ' (poisoned)'

        if title_prefix is not None:
            title = title_prefix + title

        plt.title(title)
        plt.imshow(image_data)

        if show_annotations or draw_bboxes:
            ax = plt.gca()
            ax.set_autoscale_on(False)

            anns = self.get_annotations(deleted=deleted)

            polygons = []
            colors = []
            if len(anns) > 0:
                class_ids = [a.class_id for a in anns]
                for cls in range(np.max(class_ids)+1):
                    c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
                    colors.append(c)

            list_of_used_colors_in_use_order = []

            for annotation in anns:
                c = colors[annotation.class_id]
                bbox = annotation.bbox

                if show_annotations:
                    m = annotation.get_mask()

                    img = np.ones((m.shape[0], m.shape[1], 3))

                    color_mask = np.random.random((1, 3)).tolist()[0]
                    for i in range(3):
                        img[:,:,i] = color_mask[i]

                    ax.imshow(np.dstack((img, m*0.75)))

                if draw_bboxes:
                    [bbox_x, bbox_y, bbox_w, bbox_h] = bbox
                    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y]]
                    np_poly = np.array(poly).reshape((4, 2))
                    polygons.append(Polygon(np_poly))
                    list_of_used_colors_in_use_order.append(c)

            p = PatchCollection(polygons, facecolor=list_of_used_colors_in_use_order, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=list_of_used_colors_in_use_order, linewidths=2)
            ax.add_collection(p)

        if output_filepath is not None:
            plt.savefig(output_filepath)
            plt.close(plt.gcf())
        else:
            plt.show()

    def write_image(self, filepath: str):
        # get an RGBA copy of the image
        data = self.get_image_data(as_rgb=False)
        cv2.imwrite(filepath, cv2.cvtColor(data, cv2.COLOR_RGBA2BGRA))

    @staticmethod
    def read_image(filepath: str):
        # get an RGBA copy of the image
        return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGRA2RGBA)

    def write_combined_labeled_mask(self, output_filepath: str):
        # get an RGBA copy of the image
        mask = self.get_combined_labeled_mask()
        cv2.imwrite(output_filepath, mask)

    @staticmethod
    def read_combined_labeled_mask(filepath: str):
        return cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    def save_json(self, filepath: str):
        if not filepath.endswith('.json'):
            raise RuntimeError("Expecting a file ending in '.json'")
        try:
            with open(filepath, mode='w', encoding='utf-8') as f:
                f.write(jsonpickle.encode(self, warn=True, indent=2))
        except:
            msg = 'Failed writing file "{}".'.format(filepath)
            logger.warning(msg)
            raise

    @staticmethod
    def load_json(filepath: str):
        if not os.path.exists(filepath):
            raise RuntimeError("Filepath does not exists: {}".format(filepath))
        if not filepath.endswith('.json'):
            raise RuntimeError("Expecting a file ending in '.json'")
        try:
            with open(filepath, mode='r', encoding='utf-8') as f:
                obj = jsonpickle.decode(f.read())

        except json.decoder.JSONDecodeError:
            logging.error("JSON decode error for file: {}, is it a proper json?".format(filepath))
            raise
        except:
            msg = 'Failed reading file "{}".'.format(filepath)
            logger.warning(msg)
            raise

        return obj


    # def load(self, filepath: str):
    #     self.data = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)  # in BGRA
    #     self.data = cv2.cvtColor(self.data, cv2.COLOR_BGRA2RGBA)
    #     self.mask = (self.data[:, :, 3] > 0).astype(bool)
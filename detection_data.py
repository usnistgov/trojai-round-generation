# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import copy
import logging

import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
from pycocotools import coco
from pycocotools import mask as maskUtils

logger = logging.getLogger()


class ObjectDetectionData:
    def __init__(self, image_filepath, image_id, width, height, coco_anns):
        self.image_filepath = image_filepath
        self.image_id = image_id
        self.coco_anns = coco_anns
        self.width = width
        self.height = height
        self.image_data = None
        self.compressed_image_data = None

    def get_image_data(self):
        if self.image_data is not None:
            return copy.deepcopy(self.image_data)
        # convert into normal numpy array
        image_data = cv2.imdecode(self.compressed_image_data, cv2.IMREAD_UNCHANGED)
        # cv2 loads as BGR not RGB
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

        if len(image_data.shape) > 3:
            raise Exception("Found image with >3 dimensions? pls fix.")
        elif len(image_data.shape) < 2:
            raise Exception("Found image with <2 dimensions? pls fix.")
        elif len(image_data.shape) == 2:
            image_data = np.stack((image_data, image_data, image_data))
            # move channels last to align with normally loaded images
            image_data = np.transpose(image_data, (1, 2, 0))

        return image_data

    def has_class_id(self, class_id: int):
        for ann in self.coco_anns:
            if ann['category_id'] == class_id:
                return True

        return False

    def annToRLE(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        h = self.height
        w = self.width
        segm = ann['segmentation']
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann)
        m = maskUtils.decode(rle)
        return m

    def get_all_anns_for_class(self, class_id: int):
        if class_id is None:
            valid_anns = self.coco_anns
        else:
            valid_anns = [elem for elem in self.coco_anns if elem['category_id'] == class_id]
        return valid_anns

    def find_detections(self, class_id: int, min_area: float = 0, max_area: float = float('inf')) -> list:
        valid_anns = self.get_all_anns_for_class(class_id)
        valid_anns = [elem for elem in valid_anns if min_area < elem['area'] < max_area]
        return valid_anns

    def has_image_id(self, image_id):
        return self.image_id == image_id

    def load_data(self):
        if self.compressed_image_data is not None:
            logging.warning('Called load_data multiple times, compressed_image_data is already loaded')

        # self.image_data = cv2.imread(self.image_filepath, cv2.IMREAD_UNCHANGED)

        with open(self.image_filepath, 'rb') as fh:
            self.compressed_image_data = np.fromstring(fh.read(), np.uint8)

    def view_data(self, draw_bbox=False, title=None, show=True):
        if self.compressed_image_data is None:
            self.load_data()

        image_data = self.get_image_data()

        if title is None:
            title = self.image_filepath

        plt.title(title)
        plt.imshow(image_data)

        self.showAnns(draw_bbox=draw_bbox)

        if show:
            plt.show()
        logging.info('Finished showing {}'.format(self.image_filepath))

    def showAnns(self, draw_bbox=False):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(self.coco_anns) == 0:
            return 0
        if 'segmentation' in self.coco_anns[0] or 'keypoints' in self.coco_anns[0]:
            datasetType = 'instances'
        elif 'caption' in self.coco_anns[0]:
            datasetType = 'captions'
        else:
            raise Exception('datasetType not supported')
        if datasetType == 'instances':
            import matplotlib.pyplot as plt
            from matplotlib.collections import PatchCollection
            from matplotlib.patches import Polygon

            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in self.coco_anns:
                c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # polygon
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg)/2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                    else:
                        # mask
                        if type(ann['segmentation']['counts']) == list:
                            rle = maskUtils.frPyObjects([ann['segmentation']], self.height, self.width)
                        else:
                            rle = [ann['segmentation']]
                        m = maskUtils.decode(rle)
                        img = np.ones( (m.shape[0], m.shape[1], 3) )
                        if ann['iscrowd'] == 1:
                            color_mask = np.array([2.0,166.0,101.0])/255
                        if ann['iscrowd'] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:,:,i] = color_mask[i]
                        ax.imshow(np.dstack( (img, m*0.5) ))

                if draw_bbox:
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
                    np_poly = np.array(poly).reshape((4,2))
                    polygons.append(Polygon(np_poly))
                    color.append(c)

            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)
        elif datasetType == 'captions':
            for ann in self.coco_anns:
                print(ann['caption'])
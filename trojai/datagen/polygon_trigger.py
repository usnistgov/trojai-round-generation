# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import numpy as np

import cv2
import PIL.Image
import PIL.ImageDraw
from functools import reduce
import operator
import math

from numpy.random import RandomState

import trigger_executor
from .image_entity import ImageEntity
from imgaug import augmenters as iaa


class PolygonTrigger(ImageEntity):
    """
    Defines a programmatically created polygon trigger.
    """

    def __init__(self, img_size: int, n_sides: int, random_state_obj: RandomState, color=None, texture_augmentation=None, filepath=None):
        """
        Initializes a polygon object
        :param img_size: the size of the polygons constraining rectangle.
        :param n_sides: the number of polygon sides.
        :param color: polygon color
        :param filepath: filepath, if not None, trigger will be loaded from that path
        """

        if color is None:
            raise RuntimeError('Polygon color must be "any" or a 3 elements list, not None')

        if color != "any" and len(color) != 3:
            raise RuntimeError('Polygon color must be a 3 elements list, or "any".')

        self.color = color
        self.texture_augmentation = texture_augmentation

        if filepath is not None:
            self.load(filepath)
            return

        self.data = PIL.Image.new('RGBA', (img_size, img_size))
        draw = PIL.ImageDraw.Draw(self.data)

        done = False
        while not done:
            done = True
            coords = list()
            for pt in range(n_sides):
                x = random_state_obj.randint(int(0.1 * img_size), int(0.9 * img_size))
                y = random_state_obj.randint(int(0.1 * img_size), int(0.9 * img_size))
                coords.append((x, y))

            center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
            coords = sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)

            min_x = img_size
            max_x = 0
            min_y = img_size
            max_y = 0
            for c in coords:
                min_x = min(min_x, c[0])
                max_x = max(max_x, c[0])
                min_y = min(min_y, c[1])
                max_y = max(max_y, c[1])

            delta_x = max_x - min_x
            delta_y = max_y - min_y
            if delta_x < 10 or delta_y < 10:
                done = False

        x_scale = img_size / (max_x - min_x)
        y_scale = img_size / (max_y - min_y)

        new_coords = list()
        for c in coords:
            new_x = int(x_scale * (c[0] - min_x))
            new_y = int(y_scale * (c[1] - min_y))
            new_coords.append((new_x, new_y))

        coords = new_coords
        if self.color == "any":
            draw.polygon(coords, fill=(128, 128, 128, 255))
        else:
            draw.polygon(coords, fill=(self.color[0], self.color[1], self.color[2], 255))
        # convert from PIL image to numpy array
        self.data = np.array(self.data)
        self.mask = (self.data[:, :, 3] > 0).astype(bool)
        self.area = np.count_nonzero(self.mask)

        # pre-build the texture if neither the color nor the texture needs to shift
        if self.color != "any" and self.texture_augmentation != 'any':
            # apply the texture now and save it
            self.data = self.add_texture_to_polygon(self.data, random_state_obj)

    def update_trigger_color_texture(self, rso):
        if self.color == "any":
            # pick a random color
            options = [o for o in trigger_executor.PolygonTriggerExecutor.TRIGGER_COLOR_LEVELS if o != "any"]
            idx = list(range(len(options)))
            c = options[rso.choice(idx)]
            data = np.copy(self.data)
            data[self.mask, 0:3] = c
            data = self.add_texture_to_polygon(data, rso)
            self.data = data
        else:
            if self.texture_augmentation == 'any':
                data = np.copy(self.data)
                data = self.add_texture_to_polygon(data, rso)
                self.data = data

    def get_data(self) -> np.ndarray:
        """
        Get the data associated with the ImageEntity
        :return: return a numpy.ndarray representing the image
        """
        return self.data

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the ImageEntity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

    def get_semantic_label_mask(self) -> None:
        """
        Get the semantic mask associated with the ImageEntity
        :return: None, this function does not apply to this class
        """
        return None

    def save(self, output_filepath: str):
        if self.data.shape[2] == 4:
            pass
        else:
            raise RuntimeError("Polygon trigger should be RGBA, but {} channels were found".format(self.data.shape[2]))
        cv2.imwrite(output_filepath, cv2.cvtColor(self.data, cv2.COLOR_RGBA2BGRA))

    def load(self, filepath: str):
        self.data = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)  # in BGRA
        self.data = cv2.cvtColor(self.data, cv2.COLOR_BGRA2RGBA)
        self.mask = (self.data[:, :, 3] > 0).astype(bool)

    def add_texture_to_polygon(self, data, rso: np.random.RandomState):
        # DO NOTHING if augmentation is identity

        tex_aug = self.texture_augmentation
        if tex_aug == "any":
            # support None as a random texture
            options = [o for o in trigger_executor.PolygonTriggerExecutor.POLYGON_TEXTURE_AUGMENTATION_LEVELS if o != 'any']
            tex_aug = rso.choice(options)

        if tex_aug == 'fog':
            augmentation = iaa.imgcorruptlike.Fog(severity=5)
        elif tex_aug == 'frost':
            augmentation = iaa.imgcorruptlike.Frost(severity=5)
        elif tex_aug == 'snow':
            augmentation = iaa.imgcorruptlike.Snow(severity=5)
        elif tex_aug == 'spatter':
            augmentation = iaa.imgcorruptlike.Spatter(severity=5)
        elif tex_aug == 'identity':
            # handle case where identity was picked as the random texture
            return data
        else:
            raise RuntimeError('Unknown texture augmentation option {}'.format(self.texture_augmentation))

        seq = iaa.Sequential([
            augmentation,
            iaa.CLAHE(),
            iaa.Cartoon(edge_prevalence=16.0),
        ])

        polygon_rgb = data[:, :, 0:3]
        polygon_a = data[:, :, 3:4]

        original_size = np.shape(polygon_rgb)[0:2]

        # Shrink image closer to size it will be used at before applying filters
        polygon_rgb = cv2.resize(polygon_rgb, (64, 64))
        polygon_rgb = seq(images=[polygon_rgb])[0]
        polygon_rgb = cv2.resize(polygon_rgb, original_size)

        data = np.concatenate((polygon_rgb, polygon_a), axis=2)

        # only keep the texture over the polygon, not the background
        data[self.mask == 0, :] = 0
        return data


import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

import dataset



def view_data(tensor, annotations, output_filepath, ann_output_filepath):
    image_data = tensor.detach().cpu().numpy()
    # into HWC
    image_data = np.transpose(image_data, (1, 2, 0))

    plt.imshow(image_data)


    ax = plt.gca()
    ax.set_autoscale_on(False)

    plt.savefig(output_filepath)

    boxes = annotations['boxes'].detach().cpu().numpy()
    class_ids = annotations['labels'].detach().cpu().numpy()
    anns = list()
    for i in range(boxes.shape[0]):
        a = dict()
        a['bbox'] = boxes[i,:].tolist()
        a['class_id'] = int(class_ids[i])
        anns.append(a)

    polygons = []
    colors = []
    if len(anns) > 0:
        class_ids = [a['class_id'] for a in anns]
        for cls in range(np.max(class_ids) + 1):
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            colors.append(c)

    list_of_used_colors_in_use_order = []

    for annotation in anns:
        c = colors[annotation['class_id']]
        bbox = annotation['bbox']

        [bbox_x1, bbox_y1, bbox_x2, bbox_y2] = bbox
        # poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y]]
        poly = [[bbox_x1, bbox_y1],[bbox_x1, bbox_y2],[bbox_x2, bbox_y2],[bbox_x2, bbox_y1]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        list_of_used_colors_in_use_order.append(c)

    p = PatchCollection(polygons, facecolor=list_of_used_colors_in_use_order, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=list_of_used_colors_in_use_order, linewidths=2)
    ax.add_collection(p)

    plt.savefig(ann_output_filepath)
    plt.close(plt.gcf())





coco_dataset_dirpath = "/home/mmajurski/usnistgov/trojai-round-generation-private/coco"

# train_dataset = dataset.CocoDataset(os.path.join(coco_dataset_dirpath, 'train2017'),
#                                             os.path.join(coco_dataset_dirpath, 'annotations', 'instances_train2017.json'),
#                                             load_dataset=True)

train_dataset = dataset.CocoDataset(os.path.join(coco_dataset_dirpath, 'val2017'), os.path.join(coco_dataset_dirpath, 'annotations', 'instances_val2017.json'), load_dataset=True)


train_dataset.load_image_data()
ofp = '/home/mmajurski/Downloads/r10/coco-annotations-visualized'

idx = list(range(len(train_dataset)))
np.random.shuffle(idx)
idx = idx[0:200]

for i in idx:
    dat = train_dataset[i]
    output_filepath = os.path.join(ofp, "{:08}-raw.jpg".format(i))
    ann_output_filepath = os.path.join(ofp, "{:08}-ann.jpg".format(i))
    view_data(dat[0], dat[1], output_filepath, ann_output_filepath)
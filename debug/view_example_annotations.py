import os
import numpy as np
import cv2
import json
import torch
import torchvision
import matplotlib
from matplotlib import pyplot as plt
from pycocotools import mask as maskUtils


def showAnns(coco_anns, height, width, draw_bbox=False, draw_number=False):
    """
    Display the specified annotations.
    :param anns (array of object): annotations to display
    :return: None
    """
    if len(coco_anns) == 0:
        return 0
    if 'segmentation' in coco_anns[0] or 'keypoints' in coco_anns[0]:
        datasetType = 'instances'
    elif 'caption' in coco_anns[0]:
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
        for ann in coco_anns:
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
                        rle = maskUtils.frPyObjects([ann['segmentation']], height, width)
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
            if draw_number:
                [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                cx = bbox_x + int(bbox_w / 2)
                cy = bbox_y + int(bbox_h / 2)
                #ax.text(cx, cy, "{}".format(ann['category_id']), c='k', fontsize='large', fontweight='bold')
                ax.text(cx, cy, "{}".format(ann['category_id']), c=c, backgroundcolor='white', fontsize='small', fontweight='bold')

        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)


def inference(model_filepath, image):
    # inference and use those boxes instead of the annotations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the model
    pytorch_model = torch.load(model_filepath)
    # move the model to the device
    pytorch_model.to(device)
    pytorch_model.eval()

    with torch.no_grad():
        # convert the image to a tensor
        # should be uint8 type, the conversion to float is handled later
        image = torch.as_tensor(image)
        # move channels first
        image = image.permute((2, 0, 1))
        # convert to float (which normalizes the values)
        image = torchvision.transforms.functional.convert_image_dtype(image, torch.float)
        images = [image]  # wrap into list

        images = list(image.to(device) for image in images)

        outputs = pytorch_model(images)
        outputs = outputs[0]  # get the results for the single image forward pass
    return outputs


def draw_boxes(img, boxes, colors_list=None):
    """
    Args:
        img: Image to draw boxes onto
        boxes: boxes in [x1, y1, x2, y2] format
        value: what pixel value to draw into the image

    Returns: modified image

    """
    buff = 2

    if boxes is None:
        return img

    if colors_list is None:
        # default to red
        colors_list = list()
        while len(colors_list) < boxes.shape[0]:
            colors_list.append([255, 0, 0])

    # make a copy to modify
    img = img.copy()
    for i in range(boxes.shape[0]):
        x_st = round(boxes[i, 0])
        y_st = round(boxes[i, 1])

        x_end = round(boxes[i, 2])
        y_end = round(boxes[i, 3])

        # x_st = int(round(x_st))
        # x_end = int(round(x_st + w + 1))
        # y_st = int(round(y_st))
        # y_end = int(round(y_st + h + 1))

        # draw a rectangle around the region of interest
        img[y_st:y_st+buff, x_st:x_end, :] = colors_list[i]
        img[y_end-buff:y_end, x_st:x_end, :] = colors_list[i]
        img[y_st:y_end, x_st:x_st+buff, :] = colors_list[i]
        img[y_st:y_end, x_end-buff:x_end, :] = colors_list[i]

    return img


def visualize_image(img_fp, output_dirpath, draw_bbox=False, draw_number=False, model_filepath=None):
    parent_fp, img_fn = os.path.split(img_fp)
    ann_fn = img_fn.replace('.jpg', '.json')
    ann_fp = os.path.join(parent_fp, ann_fn)

    if not os.path.exists(img_fp):
        raise RuntimeError('Requested image file {} does not exists.'.format(img_fp))
    if not os.path.exists(ann_fp):
        raise RuntimeError('Requested image annotation file {} does not exists.'.format(ann_fp))

    img = cv2.imread(img_fp, cv2.IMREAD_UNCHANGED)  # loads to BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
    height = img.shape[0]
    width = img.shape[1]

    # clear the figure
    plt.clf()

    if model_filepath is not None:
        outputs = inference(model_filepath, img)
        boxes = outputs['boxes'].detach().cpu().numpy()
        scores = outputs['scores'].detach().cpu().numpy()
        labels = outputs['labels'].detach().cpu().numpy()
        boxes = boxes[scores > 0.5, :]

        # get colors for the boxes
        colors_list = list()
        cmap = matplotlib.cm.get_cmap('jet')
        for b in range(boxes.shape[0]):
            idx = float(b) / float(boxes.shape[0])
            c = cmap(idx)
            c = c[0:3]
            c = [int(255.0 * x) for x in c]
            colors_list.append(c)

        # draw the output boxes onto the image
        img = draw_boxes(img, boxes, colors_list)
        # show the image
        plt.imshow(img)

        if draw_number:
            ax = plt.gca()

            for b in range(boxes.shape[0]):
                c = colors_list[b]
                # ax.text needs color in [0, 1]
                c = [float(x) / 255.0 for x in c]
                [bbox_x1, bbox_y1, bbox_x2, bbox_y2] = boxes[b, :]
                cx = int(float((bbox_x1 + bbox_x2) / 2.0))
                cy = int(float((bbox_y1 + bbox_y2) / 2.0))
                ax.text(cx, cy, "{}".format(labels[b]), c=c, backgroundcolor='white', fontsize='small', fontweight='bold')
    else:
        with open(ann_fp, 'r') as fh:
            coco_anns = json.load(fh)

        # show the image
        plt.imshow(img)
        # draw the annotations on top of that image
        showAnns(coco_anns, height, width, draw_bbox=draw_bbox, draw_number=draw_number)

    # save figure to disk
    if output_dirpath is not None:
        if not os.path.exists(output_dirpath):
            os.makedirs(output_dirpath)
        plt.savefig(os.path.join(output_dirpath, img_fn))
    else:
        # or render it
        plt.show()


# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(description='Visualize the example data annotations')
#     parser.add_argument('--example_dirpath', type=str, required=True, help='Filepath to the folder/directory of examples to visualize')
#     parser.add_argument('--output_dirpath', type=str, required=False, default=None, help='Filepath to the folder/directory where the outputs should be stored')
#     parser.add_argument('--draw_boxes', action='store_true', help='Whether to draw bounding boxes on visualization in addition to the segmentation mask')
#     parser.add_argument('--draw_numbers', action='store_true', help='Whether to draw class label numbers on visualization')
#
#     args = parser.parse_args()
#
#     if args.output_dirpath is not None:
#         if os.path.exists(args.output_dirpath):
#             import shutil
#             shutil.rmtree(args.output_dirpath)
#
#     ifp = args.example_dirpath
#     imgs = [os.path.join(ifp, fn) for fn in os.listdir(ifp) if fn.endswith('.jpg')]
#
#     for img in imgs:
#         visualize_image(img, args.output_dirpath, args.draw_boxes, args.draw_numbers)


if __name__ == "__main__":
    import argparse

    # parser = argparse.ArgumentParser(description='Visualize the all example data annotations')
    # parser.add_argument('--example_dirpath', type=str, required=True, help='Filepath to the folder/directory containing the TrojAI dataset')
    # parser.add_argument('--draw_boxes', action='store_true', help='Whether to draw bounding boxes on visualization in addition to the segmentation mask')
    # parser.add_argument('--draw_numbers', action='store_true', help='Whether to draw class label numbers on visualization')
    #
    # args = parser.parse_args()

    args = dict()
    args['example_dirpath'] = '/home/mmajurski/data/trojai/round10/round10-train-dataset/tmp'
    args['draw_boxes'] = True
    args['draw_numbers'] = True

    args = argparse.Namespace(**args)

    ifp = args.example_dirpath
    models = [os.path.join(ifp, fn) for fn in os.listdir(ifp) if fn.startswith('id-')]
    models.sort()

    for model in models:
        in_dir = os.path.join(ifp, model, 'poisoned-example-data')
        if not os.path.exists(in_dir): continue
        out_dir = os.path.join(ifp, model, 'poisoned-example-data-visualization')
        imgs = [os.path.join(in_dir, fn) for fn in os.listdir(in_dir) if fn.endswith('.jpg')]

        for img in imgs:
            model_filepath = os.path.join(ifp, model, 'model.pt')
            visualize_image(img, out_dir, args.draw_boxes, args.draw_numbers, model_filepath)


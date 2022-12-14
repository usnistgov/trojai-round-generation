# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')

import numpy as np
import os
import csv


def draw_boxes(img, boxes, value=0):
    buff = 2

    if boxes is None:
        return img

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
        img[y_st:y_st+buff, x_st:x_end, :] = value
        img[y_end-buff:y_end, x_st:x_end, :] = value
        img[y_st:y_end, x_st:x_st+buff, :] = value
        img[y_st:y_end, x_end-buff:x_end, :] = value

    return img


def write_boxes_from_xywhc(boxes, csv_filename):
    boxes = boxes.astype(np.int32)
    # write the header to a new file
    with open(csv_filename, 'w') as fh:
        fh.write('X,Y,W,H,C\n')

        # loop over the selected stars saving them as tiffs thumbnails
        for k in range(boxes.shape[0]):
            # get the current annotation location
            x = boxes[k, 0]
            y = boxes[k, 1]
            w = boxes[k, 2]
            h = boxes[k, 3]
            c = boxes[k, 4]
            # output the position to a csv file
            fh.write('{:d},{:d},{:d},{:d},{:d}\n'.format(x, y, w, h, c))


def write_boxes_from_ltrbc(boxes, csv_filename):
    boxes = boxes.astype(np.int32)
    # write the header to a new file
    with open(csv_filename, 'w') as fh:
        fh.write('X,Y,W,H,C\n')

        # loop over the selected stars saving them as tiffs thumbnails
        for k in range(boxes.shape[0]):
            # get the current annotation location
            x = boxes[k, 0]
            y = boxes[k, 1]
            w = boxes[k, 2] - x + 1
            h = boxes[k, 3] - y + 1
            c = boxes[k, 4]
            # output the position to a csv file
            fh.write('{:d},{:d},{:d},{:d},{:d}\n'.format(x, y, w, h, c))


def write_boxes_from_ltrbpc(boxes, csv_filename):

    # write the header to a new file
    with open(csv_filename, 'w') as fh:
        fh.write('X,Y,W,H,P,C\n')

        # loop over the selected stars saving them as tiffs thumbnails
        for k in range(boxes.shape[0]):
            # get the current annotation location
            x = int(boxes[k, 0])
            y = int(boxes[k, 1])
            w = int(boxes[k, 2] - x + 1)
            h = int(boxes[k, 3] - y + 1)
            p = boxes[k, 4]
            c = int(boxes[k, 5])
            # output the position to a csv file
            fh.write('{:d},{:d},{:d},{:d},{:f},{:d}\n'.format(x, y, w, h, p, c))


def load_boxes_to_ltrbc(filepath):
    A = []

    if os.path.exists(filepath):
        with open(filepath) as csvfile:
            csv.register_dialect("comma_and_ws", skipinitialspace=True)
            reader = csv.DictReader(csvfile, dialect="comma_and_ws")
            for row in reader:
                vec = []
                vec.append(int(row['X']))
                vec.append(int(row['Y']))
                vec.append(int(row['W']))
                vec.append(int(row['H']))
                vec.append(int(row['C']))
                vec[2] = vec[0] + vec[2] - 1  # convert from width to x_end
                vec[3] = vec[1] + vec[3] - 1  # convert from height to y_end
                A.append(vec)

    A = np.asarray(A, dtype=np.int32)
    A = A.reshape(-1, 5)
    return A


def load_boxes_to_xywhc(filepath):
    A = []

    if os.path.exists(filepath):
        with open(filepath) as csvfile:
            csv.register_dialect("comma_and_ws", skipinitialspace=True)
            reader = csv.DictReader(csvfile, dialect="comma_and_ws")
            for row in reader:
                vec = []
                vec.append(int(row['X']))
                vec.append(int(row['Y']))
                vec.append(int(row['W']))
                vec.append(int(row['H']))
                vec.append(int(row['C']))
                A.append(vec)

    A = np.asarray(A, dtype=np.int32)
    A = A.reshape(-1, 5)
    return A


def box_union(boxes, weights):
    bb = np.zeros((1,4))
    bb[0,0] = np.min(boxes[:, 0])
    bb[0,1] = np.min(boxes[:, 1])
    bb[0,2] = np.max(boxes[:, 2])
    bb[0,3] = np.max(boxes[:, 3])
    w = np.mean(weights)
    # w = np.sum(weights)
    return bb, w


def union_all_overlapping_bb(boxes, scores, minimum_iou_for_merge=0):
    if len(scores) == 0 or len(scores) == 1:
        return boxes, scores

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # boxes are [left, top, right, bottom]
    x_left = boxes[:, 0]
    y_top = boxes[:, 1]
    x_right = boxes[:, 2]
    y_bottom = boxes[:, 3]
    areas = (x_right - x_left) * (y_bottom - y_top)

    scores_indexes = scores.argsort()[::-1].tolist()
    iteration_count_since_last_change = 0
    while len(scores_indexes):
        # terminate if we only have 1 box left
        if len(scores_indexes) <= 1:
            break
        # terminate if we have run through the whole list without finding any additional merges
        if iteration_count_since_last_change > len(scores_indexes):
            break

        index = scores_indexes.pop(0) # get first element

        # compute the iou
        ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index], areas[scores_indexes])
        idx = (ious > minimum_iou_for_merge).nonzero()[0]
        filtered_indexes = set(idx)
        if len(idx) > 0:
            iteration_count_since_last_change = 0
            idx = np.array(scores_indexes)[idx]  # update numbering
            # add in the ref bounding box
            idx = np.append(idx, index)
            # union boxes together
            (new_bb, w) = box_union(boxes[idx], scores[idx])
            # print('Merged Box: [{},{},{},{}]'.format(new_bb[0][0],new_bb[0][1],new_bb[0][2],new_bb[0][3]))
            boxes[index, 0] = new_bb[0, 0]
            boxes[index, 1] = new_bb[0, 1]
            boxes[index, 2] = new_bb[0, 2]
            boxes[index, 3] = new_bb[0, 3]
            scores[index] = w
            areas[index] = (new_bb[0,2] - new_bb[0,0]) * (new_bb[0,3] - new_bb[0,1])
        else:
            iteration_count_since_last_change += 1
        # append the new box to the end of the list (it was removed with the pop(0)
        scores_indexes.append(index)

        # remove the merged boxes from the list of existing boxes
        scores_indexes = [
            v for (i, v) in enumerate(scores_indexes)
            if i not in filtered_indexes
        ]

    boxes = boxes[np.array(scores_indexes), :]
    scores = scores[np.array(scores_indexes)]
    return boxes, scores


def compute_iou(box, boxes, box_area=None, boxes_area=None):
    if box.shape[0] == 0 or boxes.shape[0] == 0:
        return np.zeros((0))
    # this is the iou of the box against all other boxes
    x_left = np.maximum(box[0], boxes[:, 0])
    y_top = np.maximum(box[1], boxes[:, 1])
    x_right = np.minimum(box[2], boxes[:, 2])
    y_bottom = np.minimum(box[3], boxes[:, 3])

    intersections = np.maximum(y_bottom - y_top, 0) * np.maximum(x_right - x_left, 0)
    if box_area is None:
        box_area = (box[2] - box[0]) * (box[3] - box[1])
    if boxes_area is None:
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    unions = box_area + boxes_area - intersections
    ious = intersections / unions
    return ious


def single_class_nms(boxes, scores, iou_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        order = order[1:]

        iou = compute_iou(boxes[i, :], boxes[order, :], areas[i], areas[order])

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds]

    return keep


def per_class_nms(boxes, objectness, class_probs, iou_threshold=0.3, score_threshold=0.1):
    # all boxes belong to the same image

    num_classes = class_probs.shape[1]
    scores = class_probs * objectness  # create blend of objectness and probs
    scores = np.sqrt(scores)  # undo the probability squaring above

    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []
    for i in range(num_classes):
        indices = np.where(scores[:, i] >= score_threshold)

        filter_boxes = boxes[indices]
        filter_scores = scores[:, i][indices]
        if len(filter_boxes) == 0:
            continue

        # do non_max_suppression on the cpu
        indices = single_class_nms(filter_boxes, filter_scores, iou_threshold=iou_threshold)
        picked_boxes.append(filter_boxes[indices])
        picked_score.append(filter_scores[indices])
        picked_label.append(np.ones(len(indices), dtype='int32') * i)

    if len(picked_boxes) == 0:
        return None, None, None

    boxes = np.concatenate(picked_boxes, axis=0)
    score = np.concatenate(picked_score, axis=0)
    label = np.concatenate(picked_label, axis=0)

    return boxes, score, label


def filter_small_boxes(boxes, min_size):
    # filter out any boxes which have a width or height < 32 pixels
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    idx = np.logical_and(w > min_size, h > min_size)
    boxes = boxes[idx, :]
    return boxes


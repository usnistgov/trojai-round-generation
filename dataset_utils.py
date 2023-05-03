import multiprocessing
from typing import Iterable, Dict, List

import numpy as np
import cv2
import logging

# local imports
import base_config
from detection_data import DetectionData
import dataset



def insert_trigger_worker(det_data: DetectionData, config: base_config.Config, rso: np.random.RandomState, class_counts: list, trojan_instances_counters: Dict[int, multiprocessing.Value]):
    """
    Worker function to insert a trojan (spurious or not) into the provided DetectionData object.

    Args:
        det_data DetectionData to attempt to insert a trigger into.
        config The RoundConfig object controlling this AI models setup and configuration.
        rso The random state object to draw randomness from.
        class_counts The number of instances of each class in the dataset.
        trojan_instances_counters A Dict of multiprocessing.Value elements, one for each trojan in the trigger list of the config. The value contains the number of trojaned images created so far during this parallel poisoned of the dataset. This value allows the triggering to only affect the requested percentage of the dataset.

    Returns:
        det_data DetectionData with its inserted trigger (spurious or not).
        attempted_trigger_executor The trigger executor that was attempted to be placed into the det_data, though trigger insertion might have failed.

    """
    attempted_trigger_executor = None

    # attempt to apply triggers in a random order
    trigger_idx_list = list(range(len(config.triggers)))
    rso.shuffle(trigger_idx_list)
    for trigger_idx in trigger_idx_list:
        trigger_executor = config.triggers[trigger_idx].value
        trigger_id = trigger_executor.trigger_id

        if not det_data.spurious and config._master_rso.uniform() < trigger_executor.spurious_trigger_fraction.value:
            if hasattr(trigger_executor, 'update_trigger_color_texture'):
                # rotate the trigger color and texture if possible (only certain trigger_executors have this function)
                trigger_executor.update_trigger_color_texture(rso)

            # TODO Create a full list of Annotation objects containing all of the spurious triggers in the detection_data
            success = trigger_executor.apply_spurious_trigger(det_data, rso)
            if success:
                attempted_trigger_executor = trigger_executor

        # Identify source Ids that need to be triggered
        counter = trojan_instances_counters[trigger_id]
        with counter.get_lock():
            value = counter.value

        perc_trojaned = float(value) / float(class_counts[trigger_executor.source_class])
        # if we have already created enough triggered data for this trigger
        # if trigger_executor.trigger_fraction is None, then there is no cap on the number of triggers
        if trigger_executor.trigger_fraction is not None and trigger_executor.trigger_fraction.value is not None:
            if perc_trojaned >= trigger_executor.trigger_fraction.value:
                continue

        if not trigger_executor.is_valid(det_data):
            continue

        if hasattr(trigger_executor, 'update_trigger_color_texture'):
            # rotate the trigger color and texture if possible (only certain trigger_executors have this function)
            trigger_executor.update_trigger_color_texture(rso)

        attempted_trigger_executor = trigger_executor
        triggered_class_id = None
        for trigger_insertion_attempt in range(dataset.IMAGE_BUILD_NUMBER_TRIES):
            triggered_class_id = trigger_executor.apply_trigger(det_data, rso)
            # if the trigger is successfully inserted, break
            if triggered_class_id is not None:
                break
        # det_data.view_data()
        if triggered_class_id is not None:
            # TODO Create a full list of Annotation objects containing all of the triggers in the detection_data
            with trojan_instances_counters[trigger_id].get_lock():
                trojan_instances_counters[trigger_id].value += 1

        if det_data.poisoned:
            # Only apply 1 trigger executor per image, so return as soon as one insertion is successful
            return det_data, attempted_trigger_executor

    return det_data, attempted_trigger_executor


def filter_small_objects(labeled_mask: np.ndarray, size: int = 50) -> (np.ndarray, bool):
    if not issubclass(labeled_mask.dtype.type, np.integer):
        raise RuntimeError("Input must be integer typed to filter small objects.")

    max_id = np.max(labeled_mask)
    completely_deleted_blob = False
    for i in range(max_id):
        mask = labeled_mask == i + 1  # +1 offset to allow the background to have id=0, and cover max_id
        mask = mask.astype(np.uint8)
        _, label_ids, stats, _ = cv2.connectedComponentsWithStats(mask, 4)
        deleted_blobs = np.zeros((stats.shape[0]), dtype=np.int)
        for k in range(stats.shape[0]):
            area = stats[k, cv2.CC_STAT_AREA]
            if area < size:
                deleted_blobs[k] = 1
                labeled_mask[label_ids == k] = 0
        # if all (or all but one (background)) blob is deleted, warn the caller
        if np.count_nonzero(deleted_blobs) >= (deleted_blobs.size - 1):
            logging.debug("Object {} from labeled_mask would be completely deleted, as all of its area was in blobs <{} pixels".format(i + 1, size))
            completely_deleted_blob = True
    return labeled_mask, completely_deleted_blob
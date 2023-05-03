import os
from abc import abstractmethod
import numpy as np
import jsonpickle
import torch.utils.data
import logging
import copy

# local imports
import base_config
import detection_data


IMAGE_BUILD_NUMBER_TRIES = 20


class ImageDataset(torch.utils.data.Dataset):
    """A class representing a :class:`ImageDataset`.

    Baseline TrojAI image task dataset wrapper around the basic PyTorch Dataset object.
    All datasets used in this codebase subclass this to provide varying behavior.

    Args:
        name (str): the name for this dataset, usually something like \"train_clean\" or \"val_poisoned\".
        config (base_config.Config): the configuration object governing this training instance.
        rso (np.random.RandomState): the random state object controlling any randomness used within this dataset
        dataset_dirpath (str): the absolute filepath to the directory containing the dataset. This is only used for non-synthetic datasets.
        augmentation_transforms: The augmentation transforms to apply within the __getitem__ function before returning the data instance.
    """

    def __init__(self, name: str, config: base_config.Config,
                 rso: np.random.RandomState,
                 dataset_dirpath: str,
                 augmentation_transforms=None,
                 nb_reps=1):
        self.name = name
        self.config = config
        self._rso = rso
        self.dataset_dirpath = dataset_dirpath
        self.augmentation_transforms = augmentation_transforms
        self.nb_reps = max(nb_reps, 1)  # controls the number of reps through the dataset for 1 epoch

        # list to store all the data instances
        self.all_detection_data = list()
        # list to store only the poisoned instances (shallow copy)
        self.all_poisoned_data = list()
        # list to store only the clean instances (shallow copy)
        self.all_clean_data = list()

    def get_class_distribution(self) -> dict[int, int]:
        """Function to get a dictionary of the number of instances per class.
        """
        class_distribution = dict()
        for c in range(self.config.number_classes.value):
            class_distribution[c] = 0
        for i in range(len(self.all_detection_data)):
            det_data = self.all_detection_data[i]
            labels = det_data.get_class_label_list()
            for c in labels:
                class_distribution[c] += 1

        return class_distribution

    def __len__(self):
        return self.nb_reps * len(self.all_detection_data)

    def __getitem__(self, item):
        # handle potential item values higher than the len, as the nb_reps could be > 1
        item = item % len(self.all_detection_data)
        return self.all_detection_data[item]

    def set_nb_reps(self, nb_reps):
        self.nb_reps = max(nb_reps, 1)

    @abstractmethod
    def build_dataset(self, class_list: list[int] = None, verify_trojan_fraction: bool = None):
        """build_dataset refers to synthetic datasets where trojaning happens during dataset creation.
        If the dataset is being loaded from disk, use load_dataset and trojan.
        Functionality wise: build_dataset = load_dataset + trojan methods.

        Args:
            class_list: the set of class labels to build the dataset for. If None, all classes are build. A subset of class labels can be useful when creating subsets of the dataset for additional testing and evaluation.
            verify_trojan_fraction: flag controlling whether to verify the trojan percentage is as requested after dataset construction is complete.
        """
        raise NotImplementedError()

    @abstractmethod
    def load_dataset(self, class_list: list[int] = None):
        """Loads a dataset from disk, populating the required object metadata elements, storing the data in memory.

        Args:
            class_list: the set of class labels to build the dataset for. If None, all classes are build. A subset of class labels can be useful when creating subsets of the dataset for additional testing and evaluation.
        """
        raise NotImplementedError()

    @abstractmethod
    def trojan(self, verify_trojan_fraction: bool = None):
        """Trojans a dataset that is resident in memory.
        This function is used in combination with load_dataset to load and trojan non-synthetic datasets.

        Args:
            verify_trojan_fraction: flag controlling whether to verify the trojan percentage is as requested after poisoning has been completed.
        """
        raise NotImplementedError()

    @abstractmethod
    def build_examples(self, output_filepath, num_samples_per_class):
        # TODO move example data creation into here?
        raise NotImplementedError()

    def set_transforms(self, new_transforms):
        """Setter to change the augmentation transforms for this dataset.

        Args:
            new_transforms: augmentation transforms to use with this dataset.
        """
        self.augmentation_transforms = new_transforms

    def train_val_test_split(self, val_fraction: float = 0.2, test_fraction: float = 0.2, shuffle: bool = True):
        """Split the current dataset into 3 chunks, (train, val, test).
        train_fraction = 1.0 - val_fraction - test_fraction

        Args:
            val_fraction: the fraction (0.0, 1.0) of the current dataset to place in the new validation dataset.
            test_fraction: the fraction (0.0, 1.0) of the current dataset to place in the new test dataset.
            shuffle: whether to shuffle the dataset before splitting the dataset.
        """

        train_fraction = 1.0 - test_fraction - val_fraction
        if train_fraction <= 0.0:
            raise RuntimeError("Train fraction too small. {}% of the data was allocated to training split, {}% to validation split, and {}% to test split.".format(int(train_fraction * 100), int(val_fraction * 100), int(test_fraction * 100)))
        logging.info("Train data fraction: {}, Validation data fraction: {}, Test data fraction: {}".format(train_fraction, val_fraction, test_fraction))

        train_dataset = type(self)('train', self.config, self._rso, self.dataset_dirpath, self.augmentation_transforms)
        val_dataset = type(self)('val', self.config, self._rso, self.dataset_dirpath, self.augmentation_transforms)
        test_dataset = type(self)('test', self.config, self._rso, self.dataset_dirpath, self.augmentation_transforms)

        idx = list(range(len(self.all_detection_data)))
        if shuffle:
            self._rso.shuffle(idx)

        idx_test = round(len(idx) * val_fraction)
        idx_val = round(len(idx) * (val_fraction + test_fraction))
        test_keys = idx[0:idx_test]
        test_keys.sort()
        val_keys = idx[idx_test:idx_val]
        val_keys.sort()
        train_keys = idx[idx_val:]
        train_keys.sort()

        for k in train_keys:
            v = self.all_detection_data[k]
            train_dataset.all_detection_data.append(v)
            if v.poisoned:
                train_dataset.all_poisoned_data.append(v)
            else:
                train_dataset.all_clean_data.append(v)

        for k in val_keys:
            v = self.all_detection_data[k]
            val_dataset.all_detection_data.append(v)
            if v.poisoned:
                val_dataset.all_poisoned_data.append(v)
            else:
                val_dataset.all_clean_data.append(v)

        for k in test_keys:
            v = self.all_detection_data[k]
            test_dataset.all_detection_data.append(v)
            if v.poisoned:
                test_dataset.all_poisoned_data.append(v)
            else:
                test_dataset.all_clean_data.append(v)

        train_dataset.class_counts = train_dataset.get_class_distribution()
        val_dataset.class_counts = val_dataset.get_class_distribution()
        test_dataset.class_counts = test_dataset.get_class_distribution()

        return train_dataset, val_dataset, test_dataset

    def clean_poisoned_split(self):
        """Split the dataset into two, one containing just the clean data, and one containing just the poisoned data.
        """
        clean_dataset = type(self)(self.name + '_clean', self.config, self._rso, self.dataset_dirpath, self.augmentation_transforms)
        poisoned_dataset = type(self)(self.name + '_poisoned', self.config, self._rso, self.dataset_dirpath, self.augmentation_transforms)

        clean_dataset.all_detection_data.extend(self.all_clean_data)
        clean_dataset.all_clean_data.extend(self.all_clean_data)
        poisoned_dataset.all_detection_data.extend(self.all_poisoned_data)
        poisoned_dataset.all_poisoned_data.extend(self.all_poisoned_data)

        return clean_dataset, poisoned_dataset

    def get_poisoned_split(self):
        """Get just the poisoned data as a dataset
        """
        poisoned_dataset = type(self)(self.name + '_poisoned', self.config, self._rso, self.dataset_dirpath, self.augmentation_transforms)

        poisoned_dataset.all_detection_data.extend(self.all_poisoned_data)
        poisoned_dataset.all_poisoned_data.extend(self.all_poisoned_data)

        return poisoned_dataset

    def serialize(self, filepath: str):
        """Serialize this dataset object to disk in a human readable format.

        Args:
            filepath: The absolute filepath to a directory where the output will be written.
        """

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        for det_data in self.all_detection_data:
            det_data = copy.deepcopy(det_data)

            image_id = det_data.image_id
            fldr = 'id-{:08d}'.format(image_id)
            ofp = os.path.join(filepath, fldr)
            if not os.path.exists(ofp):
                os.makedirs(ofp)

            det_data.write_image(os.path.join(ofp, 'img.jpg'))
            det_data.write_combined_labeled_mask(os.path.join(ofp, 'mask.tif'))
            # TODO add serialization of trigger/spurious mask and boxes. I.e. create a full list of Annotation objects containing all of the triggers in the detection_data

            # blank out mask info, since thats stored in the global mask.tif
            for a in det_data._annotations:
                a.encoded_mask = None
            # blank out the compress image data, since thats stored in the img.jpg
            det_data._compressed_image_data = None

            det_data.save_json(os.path.join(ofp, 'detection_data.json'))

        obj_to_jsonize = copy.deepcopy(self)
        obj_to_jsonize.all_detection_data = None
        obj_to_jsonize.all_poisoned_data = None
        obj_to_jsonize.all_clean_data = None
        obj_to_jsonize.config = None
        obj_to_jsonize.augmentation_transforms = None
        ofp = os.path.join(filepath, 'dataset.json')
        try:
            with open(ofp, mode='w', encoding='utf-8') as f:
                f.write(jsonpickle.encode(obj_to_jsonize, warn=True, indent=2))
        except RuntimeError as e:
            msg = 'Failed writing file "{}".'.format(filepath)
            logging.warning(msg)
            raise

    @staticmethod
    def deserialize(filepath: str, config: base_config.Config, augmentation_transforms):
        """Serialize this dataset object to disk in a human readable format.

        Args:
            filepath: The absolute filepath to a directory where the output will be written.
            config: The RoundConfig instance to load this dataset into.
            augmentation_transforms: The augmentation transforms to apply to the datasets being loaded from disk.
        """
        ds = ImageDataset.load_json(os.path.join(filepath, 'dataset.json'))
        ds.config = config
        ds.all_detection_data = list()
        ds.all_poisoned_data = list()
        ds.all_clean_data = list()
        # augmentation_transforms don't serialize properly, so they need to be passed back in
        ds.augmentation_transforms = augmentation_transforms

        fldrs = [f for f in os.listdir(filepath) if f.startswith('id-')]
        fldrs.sort()
        for f in fldrs:
            ofp = os.path.join(filepath, f)

            img = detection_data.DetectionData.read_image(os.path.join(ofp, 'img.jpg'))
            mask = detection_data.DetectionData.read_combined_labeled_mask(os.path.join(ofp, 'mask.tif'))
            det_data = detection_data.DetectionData.load_json(filepath=os.path.join(ofp, 'detection_data.json'))

            det_data.update_image_data(img)

            for a_idx in range(len(det_data._annotations)):
                a = det_data._annotations[a_idx]
                a.encoded_mask = a.encode_mask(mask == a_idx+1)
            ds.all_detection_data.append(det_data)
            if det_data.poisoned:
                ds.all_poisoned_data.append(det_data)
            else:
                ds.all_clean_data.append(det_data)

        return ds

    def dump_jpg_examples(self, clean: bool = True, n: int = 20, spurious: bool = False):
        """Utility function to write n images from the dataset to the output_dirpath of the config, to visualize what the instances of this dataset look like.

        Args:
            clean: flag indicating whether to save the clean or poisoned examples.
            n: the number of instances to save.
            spurious: flag indicating whether to save the spurious triggered instances.
        """
        fldr = '{}-clean-example-data'.format(self.name)
        dn = self.all_clean_data
        if not clean:
            fldr = '{}-poisoned-example-data'.format(self.name)
            dn = self.all_poisoned_data
            dn = [d for d in dn if not d.spurious]
        else:
            dn = [d for d in dn if not d.spurious]
        if spurious:
            dn = self.all_clean_data
            fldr = '{}-spurious-clean-example-data'.format(self.name)
            dn = [d for d in dn if d.spurious]

        if len(dn) == 0:
            # don't bother for emtpy datasets
            return

        self._rso.shuffle(dn)
        ofp = os.path.join(self.config.output_filepath, fldr)
        if os.path.exists(ofp):
            start_idx = len([fn for fn in os.listdir(ofp) if fn.endswith(".jpg")])
        else:
            start_idx = 0
        end_idx = min(n, len(dn))
        if end_idx < n:
            logging.info("dump_jpg_examples only has {} instances available, instead of the requested {}.".format(end_idx, n))

        idx_list = list(range(start_idx, end_idx))
        if len(idx_list) > 0:
            if not os.path.exists(ofp):
                os.makedirs(ofp)

        for i in idx_list:
            det_data = dn[i]
            det_data.view_data(draw_bboxes=True, output_filepath=os.path.join(ofp, 'img-{:08}.jpg'.format(det_data.image_id)))


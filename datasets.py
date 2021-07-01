# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import os
from typing import Union
import numpy as np

import pandas as pd
from numpy.random import RandomState
import lmdb
from formats_pb2 import ImageNumberNumberTuple

from trojai.modelgen.data_descriptions import CSVImageDatasetDesc, CSVTextDatasetDesc
from trojai.modelgen.datasets import DatasetInterface


class LMDBDataset(DatasetInterface):
    """
    Defines a dataset that is represented by a CSV file with columns "key_str", "train_label", and optionally
    "true_label". The data is loaded from a specified lmdb file containing a serialized key value store of the actual data.
    "train_label" refers to the label with which the data should be trained.  "true_label" refers to the actual
    label of the data point, and can differ from train_label if the dataset is poisoned.  A CSVDataset can support
    any underlying data that can be loaded on the fly and fed into the model (for example: image data)
    """
    def __init__(self, path_to_data: str, csv_filename:str, lmdb_filename: str, true_label=False, shuffle=False,
                 random_state: Union[int, RandomState]=None,
                 data_transform=lambda x: x, label_transform=lambda l: l):
        """
        Initializes a LMDBDataset object.
        :param path_to_data: the root folder where the data lives
        :param csv_filename: the CSV file specifying the actual data points
        :param true_label (bool): if True, then use the column "true_label" as the label associated with each
        datapoint.  If False (default), use the column "train_label" as the label associated with each datapoint
        :param shuffle: if True, the dataset is shuffled before loading into the model
        :param random_state: if specified, seeds the random sampler when shuffling the data
        :param data_transform: a callable function which is applied to every data point before it is fed into the
            model. By default, this is an identity operation
        :param label_transform: a callable function which is applied to every label before it is fed into the model.
            By default, this is an identity operation.
        """
        super().__init__(path_to_data)

        self.csv_filepath = os.path.join(self.path_to_data, csv_filename)
        self.lmdb_filepath = os.path.join(self.path_to_data, lmdb_filename)

        self.true_label = true_label
        self.label = 'train_label'
        if true_label:
            self.label = 'true_label'
        self.data_df = pd.read_csv(self.csv_filepath)
        if shuffle:
            self.data_df = self.data_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        self.data_transform = data_transform
        self.label_transform = label_transform

        # set the data description
        num_classes = len(self.data_df[self.label].unique())
        self.data_description = CSVImageDatasetDesc(len(self.data_df), shuffle, num_classes)

        # open the lmdb environment for reading. This will get cleaned up on garbage collect of this object
        self.lmdb_env = lmdb.open(self.lmdb_filepath, map_size=int(2e10))
        self.lmdb_txn = self.lmdb_env.begin(write=False) # this supports multi-threaded read

    def __getitem__(self, item):
        datum = ImageNumberNumberTuple()
        key_str = self.data_df.iloc[item]["file"]
        # extract the serialized image from the database
        value = self.lmdb_txn.get(key_str.encode('ascii'))
        # convert from serialized representation
        datum.ParseFromString(value)

        # convert from string to numpy array
        data = np.fromstring(datum.image, dtype=datum.img_type)
        # reshape the numpy array using the dimensions recorded in the datum
        data = data.reshape((datum.img_height, datum.img_width, datum.channels))

        train_label = datum.train_label
        true_label = datum.true_label

        if self.true_label:
            label = true_label
        else:
            label = train_label

        # data = torch.from_numpy(data).float()
        data = self.data_transform(data)
        label = self.label_transform(label)

        return data, label

    def __len__(self):
        return len(self.data_df)

    def get_data_description(self):
        return self.data_description

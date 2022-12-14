import os
import json
import tarfile
from urllib import request

import argparse
import logging
logger = logging.getLogger()


def download_and_extract_imdb(output_filepath: str):
    """
    Downloads imdb dataset from https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz and unpacks it into
        combined path of the given top level directory and the data folder name.
    :param output_filepath: (str) folder to download into, and extract tarball in.
    """
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)

    tar_file = os.path.join(output_filepath, 'aclimdb.tar.gz')
    request.urlretrieve(url, tar_file)
    try:
        tar = tarfile.open(tar_file)
        tar.extractall(output_filepath)
        tar.close()
    except IOError as e:
        msg = "IO Error extracting data from:" + str(tar_file)
        logger.exception(msg)
        raise IOError(e)
    os.remove(tar_file)


def load_dataset(input_path):

    fns = [fn for fn in os.listdir(input_path) if fn.endswith('.txt')]
    fns.sort()
    entities = list()

    for fn in fns:
        with open(os.path.join(input_path, fn), 'r') as fo:
            entities.append(str(fo.read().replace('\n', '')))
    return entities, fns


def load_and_convert_to_json(records: dict, id: int, input_folder: str):

    # Create positive sentiment data
    input_test_pos_path = os.path.join(input_folder, 'pos')
    pos_entities, pos_filenames = load_dataset(input_test_pos_path)
    for pos_entity in pos_entities:
        key = '{:012d}'.format(id)
        id = id + 1
        entry = dict()
        entry['data'] = pos_entity
        entry['label'] = 1
        records[key] = entry

    # Create negative sentiment data
    input_test_neg_path = os.path.join(input_folder, 'neg')
    neg_entities, neg_filenames = load_dataset(input_test_neg_path)
    for neg_entity in neg_entities:
        key = '{:012d}'.format(id)
        id = id + 1
        entry = dict()
        entry['data'] = neg_entity
        entry['label'] = 0
        records[key] = entry

    return records, id


def convert_dataset(input_base_path, output_filepath):
    """
    Creates a clean dataset in a path from the raw IMDB data
    """

    records = dict()
    id = 0

    # TEST DATA
    print('Converting Test data')
    input_data_path = os.path.join(input_base_path, 'test')
    records, id = load_and_convert_to_json(records, id, input_data_path)

    # Training DATA
    print('Converting Train data')
    input_data_path = os.path.join(input_base_path, 'train')
    records, id = load_and_convert_to_json(records, id, input_data_path)

    records = list(records.values())  # convert to a list without the keys
    records = {'data': records}  # make the records compatible with the huggingface dataset loader

    with open(output_filepath, 'w') as output_file:
        json.dump(records, output_file, ensure_ascii=True, indent=2)



parser = argparse.ArgumentParser(description='Download and convert imdb dataset into HuggingFace json format.')

parser.add_argument('--download_folder', type=str, required=True,
                    help='Where to store the downloaded files.')

parser.add_argument('--output_filepath', type=str, required=True,
                    help='Output filepath to save the resulting json file at.')

args = parser.parse_args()
output_filepath = args.output_filepath
download_folder = args.download_folder

download_and_extract_imdb(download_folder)
convert_dataset(os.path.join(download_folder, 'aclImdb'), output_filepath)
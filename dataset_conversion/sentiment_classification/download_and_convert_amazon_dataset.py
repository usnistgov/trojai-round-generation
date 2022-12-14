import os
from urllib import request
import gzip
import shutil
import json

import argparse
import logging
logger = logging.getLogger()

# https://nijianmo.github.io/amazon/index.html#subsets
# @inproceedings{ni2019justifying,
#   title={Justifying recommendations using distantly-labeled reviews and fine-grained aspects},
#   author={Ni, Jianmo and Li, Jiacheng and McAuley, Julian},
#   booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
#   pages={188--197},
#   year={2019}
# }

# download urls
URLS = [
# 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/AMAZON_FASHION_5.json.gz',
'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/All_Beauty_5.json.gz',
'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Appliances_5.json.gz',
'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Arts_Crafts_and_Sewing_5.json.gz',
# 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Automotive_5.json.gz', # removed to reduce dataset size
# 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Books_5.json.gz',  # removed to reduce dataset size
# 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/CDs_and_Vinyl_5.json.gz', # removed to reduce dataset size
'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Cell_Phones_and_Accessories_5.json.gz',
# 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Clothing_Shoes_and_Jewelry_5.json.gz', # removed to reduce dataset size
'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Digital_Music_5.json.gz',
# 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Electronics_5.json.gz', # removed to reduce dataset size
# 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Gift_Cards_5.json.gz',
'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Grocery_and_Gourmet_Food_5.json.gz',
# 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Home_and_Kitchen_5.json.gz', # removed to reduce dataset size
'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Industrial_and_Scientific_5.json.gz',
# 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Kindle_Store_5.json.gz', # removed to reduce dataset size
'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Luxury_Beauty_5.json.gz',
# 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Magazine_Subscriptions_5.json.gz',
# 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Movies_and_TV_5.json.gz', # removed to reduce dataset size
'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Musical_Instruments_5.json.gz',
'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Office_Products_5.json.gz',
'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Patio_Lawn_and_Garden_5.json.gz',
# 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Pet_Supplies_5.json.gz', # removed to reduce dataset size
'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Prime_Pantry_5.json.gz',
'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Software_5.json.gz',
# 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Sports_and_Outdoors_5.json.gz', # removed to reduce dataset size
# 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Tools_and_Home_Improvement_5.json.gz', # removed to reduce dataset size
# 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Toys_and_Games_5.json.gz',# removed to reduce dataset size
'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Video_Games_5.json.gz'
]


def download_and_extract(output_folder: str, download_url: str):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fldr, fn = os.path.split(download_url)

    print("Downloading: {}".format(fn))
    local_filepath = os.path.join(output_folder, fn)
    request.urlretrieve(download_url, local_filepath)

    local_file_extracted = os.path.join(output_folder, fn.replace('.json.gz', '.json'))

    print("Extracting: {}".format(fn))
    with gzip.open(local_filepath, 'rb') as f_in:
        with open(local_file_extracted, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def load_and_convert_to_json(records: dict, id: int, input_filepath: str):
    with open(input_filepath, mode='r', encoding='utf-8') as f:
        for line in f:
            val = json.loads(line)

            if 'reviewText' not in val:
                # skip reviews without text
                continue

            if val['overall'] >= 4:
                label = 1
            elif val['overall'] <= 2:
                label = 0
            else:
                # ignore the neutral reviews
                label = None

            if label is not None:
                key = '{:012d}'.format(id)
                id += 1

                entry = dict()
                entry['data'] = str(val['reviewText'])
                entry['label'] = int(label)
                records[key] = entry

    return records, id


def convert_dataset(input_base_path, output_filepath):
    records = dict()
    id = 0

    fns = [fn for fn in os.listdir(input_base_path) if fn.endswith('.json')]
    for i in range(len(fns)):
        fn = fns[i]
        print('{}/{} : {}'.format(i, len(fns), fn))
        input_json_filepath = os.path.join(input_base_path, fn)
        records, id = load_and_convert_to_json(records, id, input_json_filepath)

    records = list(records.values())  # convert to a list without the keys

    print('dataset has {} records'.format(len(records)))

    records = {'data': records}  # make the records compatible with the huggingface dataset loader

    with open(output_filepath, 'w') as output_file:
        json.dump(records, output_file, ensure_ascii=True, indent=2)



parser = argparse.ArgumentParser(description='Download and convert the amazon dataset into HuggingFace json format.')

parser.add_argument('--download_folder', type=str, required=True,
                    help='Where to store the downloaded files.')

parser.add_argument('--output_filepath', type=str, required=True,
                    help='Output filepath to save the resulting json file at.')

args = parser.parse_args()
output_filepath = args.output_filepath
download_folder = args.download_folder

# for url in URLS:
#     download_and_extract(download_folder, url)
convert_dataset(download_folder, output_filepath)

import os
import json
import jsonpickle
import numpy as np

# https://nijianmo.github.io/amazon/index.html
# @inproceedings{ni2019justifying,
#   title={Justifying recommendations using distantly-labeled reviews and fine-grained aspects},
#   author={Ni, Jianmo and Li, Jiacheng and McAuley, Julian},
#   booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
#   pages={188--197},
#   year={2019}
# }


ifp = 'data'
ofp = 'datasets'
train_fraction = 0.8

fns = [fn for fn in os.listdir(ifp) if fn.endswith('.json')]
fns.sort()

for fn in fns:
    ofn = 'amazon-' + fn.replace('.json','')
    cur_ofp = os.path.join(ofp, ofn)
    if os.path.exists(cur_ofp):
        continue

    print(fn)

    # load the data
    data_json_filepath = os.path.join(ifp, fn)

    train_nb = 0
    test_nb = 0
    train_dict = dict()
    test_dict = dict()
    review_dict = dict()

    with open(data_json_filepath, mode='r', encoding='utf-8') as f:
        for line in f:
            val = json.loads(line)

            if 'reviewText' not in val:
                # skip reviews without text
                continue

            if int(val['overall']) not in review_dict:
                review_dict[int(val['overall'])] = 0
            review_dict[int(val['overall'])] = review_dict[int(val['overall'])] + 1

            if val['overall'] >= 4:
                label = 1
            elif val['overall'] <= 2:
                label = 0
            else:
                # ignore the neutral reviews
                label = None

            if np.random.rand() < train_fraction:
                key = '{:012d}'.format(train_nb)
                train_nb = train_nb + 1

                if label is not None:
                    entry = dict()
                    entry['data'] = str(val['reviewText'])
                    entry['label'] = int(label)
                    train_dict[key] = entry

            else:
                key = '{:012d}'.format(test_nb)
                test_nb = test_nb + 1

                if label is not None:
                    entry = dict()
                    entry['data'] = str(val['reviewText'])
                    entry['label'] = int(label)
                    test_dict[key] = entry

    if len(train_dict) == 0:
        raise RuntimeError('Train dict is empty')
    if len(test_dict) == 0:
        raise RuntimeError('Test dict is empty')

    if not os.path.exists(cur_ofp):
        os.makedirs(cur_ofp)

    with open(os.path.join(cur_ofp, 'test.json'), mode='w', encoding='utf-8') as f:
        f.write(jsonpickle.encode(test_dict, warn=True, indent=2))
    del test_dict
    with open(os.path.join(cur_ofp, 'train.json'), mode='w', encoding='utf-8') as f:
        f.write(jsonpickle.encode(train_dict, warn=True, indent=2))
    del train_dict

    print('review counts: ')
    print(review_dict)
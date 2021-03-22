import os
import jsonpickle
import random

ifp = '/mnt/scratch/trojai/data/source_data/sentiment-classification'
ofp = '/mnt/scratch/trojai/data/source_data/balanced-sentiment-classification'

datasets = [fn for fn in os.listdir(ifp) if 'amazon' in fn]
datasets.sort()

for dataset in datasets:
    print(dataset)

    for split in {'train','test'}:
        data_json_filepath = os.path.join(ifp, dataset, split + '.json')
        with open(data_json_filepath, mode='r', encoding='utf-8') as f:
            json_data = jsonpickle.decode(f.read())

        pos_sentiment_keys = list()
        neg_sentiment_keys = list()
        keys = list(json_data.keys())
        for key in keys:
            if json_data[key]['label']:
                pos_sentiment_keys.append(key)
            else:
                neg_sentiment_keys.append(key)

        random.shuffle(pos_sentiment_keys)
        random.shuffle(neg_sentiment_keys)

        n_to_select = len(pos_sentiment_keys)
        if len(neg_sentiment_keys) < n_to_select:
            n_to_select = len(neg_sentiment_keys)
        print('    selecting {} data points'.format(2*n_to_select))

        pos_sentiment_keys = pos_sentiment_keys[0:n_to_select]
        neg_sentiment_keys = neg_sentiment_keys[0:n_to_select]

        subset_json_data = dict()
        for key in pos_sentiment_keys:
            subset_json_data[key] = json_data[key]
        for key in neg_sentiment_keys:
            subset_json_data[key] = json_data[key]

        if not os.path.exists(os.path.join(ofp, dataset)):
            os.makedirs(os.path.join(ofp, dataset))

        with open(os.path.join(ofp, dataset, split + '.json'), mode='w', encoding='utf-8') as f:
            f.write(jsonpickle.encode(subset_json_data, warn=True, indent=2))


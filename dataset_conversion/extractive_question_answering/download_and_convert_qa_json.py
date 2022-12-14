import datasets
import json
from dataset_conversion.extractive_question_answering import dataset_cleaner_utils


# dataset_names = ['squad_v2', 'subjqa']
dataset_names = ['subjqa']
dataset_splits = {'squad_v2': ['train', 'validation'], 'subjqa' : ['train', 'validation', 'test']}
dataset_domains = {'squad_v2' : [], 'subjqa' : ['books', 'electronics', 'grocery', 'movies', 'restaurants', 'tripadvisor'] }

for dataset_name in dataset_names:
    splits = dataset_splits[dataset_name]
    domains = dataset_domains[dataset_name]

    all_datasets = list()
    for split in splits:
        if len(domains) > 0:
            for domain in domains:
                ds = datasets.load_dataset(dataset_name, domain, split=split, keep_in_memory=True)
                all_datasets.append(ds)
        else:
            ds = datasets.load_dataset(dataset_name,
                               split=split,
                               keep_in_memory=True)
        
            all_datasets.append(ds)
        
    dataset = datasets.concatenate_datasets(all_datasets)

    # clean up dataset to match squad answers
    def cleanup_answers(example):
        answers = example['answers']
        example['answers'] = {'answer_start': answers['answer_start'], 'text': answers['text']}
        return example

    dataset = dataset.map(cleanup_answers, keep_in_memory=True)
    
    # check for duplicates and remove them
    duplicate_index = list()
    example_id_to_index = dict()
    for i, k in enumerate(dataset['id']):
        if k in example_id_to_index.keys():
            duplicate_index.append(i)
        else:
            example_id_to_index[k] = i

    def remove_duplicates(example, idx):
        if idx in duplicate_index:
            return False
        else:
            return True

    dataset = dataset.filter(remove_duplicates, keep_in_memory=True, with_indices=True)

    dataset = dataset.map(dataset_cleaner_utils.remove_answer_not_found_subjqa,
                          num_proc=1,
                          keep_in_memory=True)

    dataset = dataset.filter(dataset_cleaner_utils.check_context_answer_alignment,
                             num_proc=1,
                             keep_in_memory=True)

    # ensure that the question, context, and answer do not have double spaces between words
    # Note this call to remove_example_duplicate_whitespace cannot be batched as the function doens't handle that data type
    dataset = dataset.map(dataset_cleaner_utils.remove_example_duplicate_whitespace,
                          num_proc=1,
                          keep_in_memory=True)

    dataset = dataset.filter(dataset_cleaner_utils.is_letter_present_in_answer_text,
                             num_proc=1,
                             keep_in_memory=True)


    dataset = dataset.map(dataset_cleaner_utils.check_errors,
                          num_proc=1,
                          keep_in_memory=True)

    print('dataset is {} examples'.format(len(dataset)))

    records = list()

    for i in range(len(dataset)):
        record = dataset[i]
        records.append(record)

    records = {'data': records}  # make the records compatible with the huggingface dataset loader
    # dataset = datasets.load_dataset('json', data_files=[fp], field='data', keep_in_memory=True, split='train')
    with open('{}.json'.format(dataset_name), 'w') as fh:
        json.dump(records, fh, ensure_ascii=True, indent=2)
        print('Finished saving dataset file: {}'.format(dataset_name))



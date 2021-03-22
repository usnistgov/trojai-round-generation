
import os
import torch
import transformers

import round_config

emb_ofp = '/mnt/scratch/trojai/data/round5/embeddings'
if not os.path.exists(emb_ofp):
    os.makedirs(emb_ofp)
tok_ofp = '/mnt/scratch/trojai/data/round5/tokenizers'
if not os.path.exists(tok_ofp):
    os.makedirs(tok_ofp)

for embedding_name in round_config.RoundConfig.EMBEDDING_LEVELS:
    print(embedding_name)
    
    # handle potential multiple flavors of embeddings
    flavors = round_config.RoundConfig.EMBEDDING_FLAVOR_LEVELS[embedding_name]
    for flavor in flavors:
        tokenizer = None
        embedding = None
        if embedding_name == 'BERT':
            tokenizer = transformers.BertTokenizer.from_pretrained(flavor)
            embedding = transformers.BertModel.from_pretrained(flavor)
        elif embedding_name == 'GPT-2':
            # ignore missing weights warning
            # https://github.com/huggingface/transformers/issues/5800
            # https://github.com/huggingface/transformers/pull/5922
            tokenizer = transformers.GPT2Tokenizer.from_pretrained(flavor)
            embedding = transformers.GPT2Model.from_pretrained(flavor)
        elif embedding_name == 'DistilBERT':
            tokenizer = transformers.DistilBertTokenizer.from_pretrained(flavor)
            embedding = transformers.DistilBertModel.from_pretrained(flavor)
        else:
            raise RuntimeError('Invalid Embedding Type: {}'.format(embedding_name))

        embedding.to('cpu')
        embedding.eval()

        torch.save(tokenizer, os.path.join(tok_ofp, '{}-{}.pt'.format(embedding_name, flavor)))
        torch.save(embedding, os.path.join(emb_ofp, '{}-{}.pt'.format(embedding_name, flavor)))
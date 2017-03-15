import numpy as np
from random import choice
import os


def are_valid_model_files(parser, path):
    args_file = path + '.model_args'
    weights_file = path + '.h5py'
    if not os.path.isfile(args_file):
        parser.error("The file %s does not exist!" % args_file)
    elif not os.path.isfile(weights_file):
        parser.error("The file %s does not exist!" % weights_file)
    else:
        return args_file, weights_file


def is_valid_file(parser, filename):
    if not os.path.isfile(filename):
        parser.error("The file %s does not exist!" % filename)
    else:
        return filename


def read_embeddings(emb_file, words):
    if len(words) == 0:
        raise ValueError('No words given to load embeddings')
    word_to_idx = dict()
    embeddings = list()
    # punctuation, prot1, prot2, is_ending
    flags = [0, 0, 0, 0]
    with open(emb_file, 'r') as f:
        for line in f:
            line = line.rstrip().split(' ')
            word, vector = line[0], line[1:]
            vector = [float(n) for n in vector]
            # space for the flags
            vector.extend(flags)
            # if word does not occur in any story, no need to get its embedding; saves a lot memory
            if word not in words:
                continue
            # index 0 cannot be used, because of masking in the embedding layer
            word_to_idx[word] = len(word_to_idx)+1
            embeddings.append(vector)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    embedding_dim = embeddings.shape[1]
    # add dummy vector for masked index 0
    embeddings = np.vstack([[0 for _ in range(embedding_dim)], embeddings])

    # insert flag embeddings
    word_to_idx['.'] = len(word_to_idx) + 1
    period_embedding = [0 for _ in range(embedding_dim)]
    period_embedding[embedding_dim-4] = 1

    word_to_idx['_PROT0_'] = len(word_to_idx) + 1
    prot0_embedding = [0 for _ in range(embedding_dim)]
    prot0_embedding[embedding_dim-3] = 1

    word_to_idx['_PROT1_'] = len(word_to_idx) + 1
    prot1_embedding = [0 for _ in range(embedding_dim)]
    prot1_embedding[embedding_dim-2] = 1

    word_to_idx['_START_ENDING_'] = len(word_to_idx) + 1
    start_ending_embedding = [0 for _ in range(embedding_dim)]
    start_ending_embedding[embedding_dim-1] = 1

    embeddings = np.vstack([embeddings, period_embedding, prot0_embedding, prot1_embedding, start_ending_embedding])

    return word_to_idx, embeddings, embedding_dim


def switch_protagonists(*story_parts):
    # 50:50 chance of switching _PROT1_ and _PROT2_ for given story parts
    do_switch = choice([True, False])
    if do_switch:
        switch_dict = dict(_PROT0_='_PROT1_', _PROT1_='_PROT0_')
        for story_part in story_parts:
            for word_id in range(len(story_part)):
                word = story_part[word_id]
                if word in switch_dict:
                    story_part[word_id] = switch_dict[word]
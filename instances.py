from random import choice
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from utils import switch_protagonists
from random import shuffle


def sentence_to_index(sentence, word_to_idx):
    index_sequence = list()
    for word in sentence:
        if word in word_to_idx:
            index_sequence.append(word_to_idx[word])
    return index_sequence


def indexify(story_ids, contexts, right_endings, wrong_endings, word_to_idx, max_input_len):
    """
    Replace words by index with vocabulary, to access the right embeddings during training or inference.
    Calculate maximum input length for context and endings; Needed to zero-pad the inputs.
    :param story_ids: list of story ids
    :param contexts: (dict) story_id -> context words
    :param right_endings: (dict) story_id -> right ending words
    :param wrong_endings: (dict) story_id -> wrong ending words
    :param word_to_idx: (dict) word -> idx
    :param max_input_len: (dict) current maximum lengths for context and endings
    :return: inplace
    """

    for story_id in story_ids:

        index_sequence = sentence_to_index(contexts[story_id], word_to_idx)
        context = np.asarray(index_sequence)
        contexts[story_id] = context

        index_sequence = sentence_to_index(right_endings[story_id], word_to_idx)
        right_ending = np.asarray(index_sequence)
        right_endings[story_id] = right_ending

        # some stories have no wrong ending yet
        if story_id in wrong_endings:
            index_sequence = sentence_to_index(wrong_endings[story_id], word_to_idx)
            wrong_ending = np.asarray(index_sequence)
            wrong_endings[story_id] = wrong_ending

    # check max length of model inputs to define maxlen parameter for zero-padding
    max_context_len = max([0] + [c.shape[0] for c in contexts.values()])
    max_ending_len = max([0] + [e.shape[0] for e in right_endings.values()])
    if len(wrong_endings) > 0:
        max_ending_len = max(max_ending_len, max([e.shape[0] for e in wrong_endings.values()]))
    max_input_len['context'] = max(max_input_len['context'], max_context_len)
    max_input_len['ending'] = max(max_input_len['ending'], max_ending_len)


def get_train_instances(story_ids, contexts, right_endings, wrong_endings, max_context_len, max_ending_len,
                        neg_sampling=5, batch_size=128):
    """
    Get the indexifyed inputs to generate instances from them. Mainly samples wrong endings for given contexts and
    build keras readable numpy arrays with them. Provides new samples in a random, infinite loop for the fit_generator
    method.
    :param story_ids: list of story ids
    :param contexts: (dict) story_id -> indices of context words
    :param right_endings: (dict) story_id -> indices of right ending words
    :param wrong_endings: (dict) story_id -> indices of wrong ending words
    :param max_context_len: number of words in the longest context
    :param max_ending_len: number of words in the longest ending
    :param neg_sampling: number of wrong ending instances to be sampled
    :param batch_size: batch_size during training
    :return: yield instances in an random, infinite loop
    """
    while True:
        # make sure fit_generator sees the instances in random order
        shuffle(story_ids)

        context_inputs = list()
        ending_inputs = list()
        label_inputs = list()

        for story_id in story_ids:
            # enough instances for this batch
            if len(context_inputs) > batch_size:
                break

            # generate right ending instance
            context = contexts[story_id]
            right_ending = right_endings[story_id]
            # by chance switch prot1 and prot2 tokens to get more variants of input
            switch_protagonists(context, right_ending)

            context_inputs.append(context)
            ending_inputs.append(right_ending)
            label_inputs.append(1)

            # generate wrong ending instance
            if story_id in wrong_endings:
                wrong_ending = wrong_endings[story_id]

                # by chance switch prot1 and prot2 tokens to get more variants of input
                switch_protagonists(wrong_ending)

                context_inputs.append(context)
                ending_inputs.append(wrong_ending)
                label_inputs.append(0)
                neg_sampling -= 1

            # generate further wrong endings (neg_sampling) by randomly selecting a right ending from another story
            for _ in range(neg_sampling):
                rdm_story_id = story_id
                while rdm_story_id == story_id:
                    rdm_story_id = choice(story_ids)
                wrong_ending = right_endings[rdm_story_id]
                # by chance switch prot1 and prot2 tokens to get more variants of input
                switch_protagonists(wrong_ending)

                context_inputs.append(context)
                ending_inputs.append(wrong_ending)
                label_inputs.append(0)

        # make sure batch is exactly the size of batch_size; may cut some neg sampled ending instances
        context_inputs = context_inputs[:batch_size]
        ending_inputs = ending_inputs[:batch_size]
        label_inputs = label_inputs[:batch_size]

        # zero pad inputs; needed for recurrent networks in keras
        context_inputs = pad_sequences(context_inputs, maxlen=max_context_len, truncating='post', padding='post')
        ending_inputs = pad_sequences(ending_inputs, maxlen=max_ending_len, truncating='post', padding='post')
        story_inputs = np.hstack([context_inputs, ending_inputs])
        label_inputs = np.asarray(label_inputs, dtype=np.int32)

        yield [story_inputs], label_inputs

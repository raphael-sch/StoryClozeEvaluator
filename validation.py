from keras.preprocessing.sequence import pad_sequences
import numpy as np
from random import shuffle
from keras.callbacks import Callback


class ValidationCallback(Callback):
    """
    Validates current model during training. Chooses the ending with the higher cosine similarity to the context as
    right ending.
    """

    def __init__(self, story_ids, contexts, right_endings, wrong_endings, max_context_len, max_ending_len, per_epochs,
                 train_logger):
        super(ValidationCallback, self).__init__()

        self.validation_data = get_validation_data(story_ids, contexts, right_endings, wrong_endings,
                                                   max_context_len, max_ending_len)
        self.per_epochs = per_epochs
        self.max_context_len = max_context_len
        self.train_logger = train_logger

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.per_epochs == 0:
            value = run_validation(self.model, self.validation_data)
            self.train_logger.write_validation(epoch, value)


def run_validation(model, validation_data, batch_size=128):
    story_inputs = validation_data
    num_stories = story_inputs.shape[0]

    # get predictions from current model
    predictions = model.predict([story_inputs], batch_size=batch_size)

    rights_cos_sims = list()
    wrong_cos_sims = list()
    y_pred_correct = 0

    # even indices are the right endings, odd the wrong ones
    for idx_right, idx_wrong in zip(range(0, num_stories, 2), range(1, num_stories, 2)):
        right_cos = predictions[idx_right][0]
        wrong_cos = predictions[idx_wrong][0]
        rights_cos_sims.append(right_cos)
        wrong_cos_sims.append(wrong_cos)

        # if right ending and wrong ending have the same cosine distance to context, select right ending randomly
        if right_cos == wrong_cos:
            rmd_labels = [1, 0]
            shuffle(rmd_labels)
            right_cos, wrong_cos = rmd_labels
        # if the model assigns the right ending a higher cosine similarity, then this instance was predicted correctly
        if right_cos > wrong_cos:
            y_pred_correct += 1
    print('')
    print('Validation')
    # average cosine similarity for the instances with the right ending
    print('avg right cosine sim: ' + str(sum(rights_cos_sims) / num_stories))
    # average cosine similarity for the instances with the wrong ending
    print('avg wrong cosine sim: ' + str(sum(wrong_cos_sims) / num_stories))
    # difference of these to averages; the higher the better
    print('avg right - avg wrong: ' + str(sum(rights_cos_sims) / num_stories - sum(wrong_cos_sims) / num_stories))
    valid_result = y_pred_correct / float(num_stories/2)
    # fraction of correctly predicted instances
    print('validation result: ' + str(valid_result))
    print('')
    return valid_result


def get_validation_data(story_ids, contexts, right_endings, wrong_endings, max_context_len, max_ending_len):
    context_inputs = list()
    ending_inputs = list()
    for story_id in story_ids:
        context_inputs.append(contexts[story_id])
        ending_inputs.append(right_endings[story_id])

        context_inputs.append(contexts[story_id])
        ending_inputs.append(wrong_endings[story_id])

    # zero pad
    context_inputs = pad_sequences(context_inputs, maxlen=max_context_len, truncating='post', padding='post')
    ending_inputs = pad_sequences(ending_inputs, maxlen=max_ending_len, truncating='post', padding='post')
    story_inputs = np.hstack([context_inputs, ending_inputs])
    return story_inputs

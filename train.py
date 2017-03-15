import argparse
from utils import is_valid_file
import random
random.seed(1337)


def run():
    """
    Read command line arguments and set defaults.
    :return: run the workflow
    """

    parser = argparse.ArgumentParser(description='Start training a model for the Story Cloze Test')

    parser.add_argument('train_file', type=lambda x: is_valid_file(parser, x),
                        help='Train the model on the stories of this file. '
                             'Can provides only right ending or additional wrong ending. \n'
                             ' In ".coref" format. See "scripts/coreference/"')

    parser.add_argument('embeddings_file', type=lambda x: is_valid_file(parser, x),
                        help='Embedding file in word2vec format.')

    parser.add_argument('-output_files', type=str, default='./model/size{lstm_size}_neg{neg_sampling}',
                        help='Basename of the output files. "x.h5py" and "x.model_args" will be created.\n'
                             'Other arguments can be accessed by {argument}. See default as example')

    parser.add_argument('-valid_file', type=lambda x: is_valid_file(parser, x),
                        help='Validate the model while training on the stories of this file. '
                             'Must include right and wrong ending. \n'
                             ' In ".coref" format. See "scripts/coreference/"')

    parser.add_argument('-lstm_size', type=int, default=300,
                        help='Size of the lstm output layer.')

    parser.add_argument('-neg_sampling', type=int, default=1,
                        help='Number of negative (wrong) endings sampled (from other right endings) per story')

    parser.add_argument('-max_epochs', type=int, default=5000,
                        help='Maximal number of epochs to train for (also valid for early stopping).')

    parser.add_argument('-patience', type=int, default=300,
                        help='Early stopping patience; Number of epochs with no improvement after which training will '
                             'be stopped.')

    parser.add_argument('-batch_size', type=int, default=128,
                        help='Batch size while training.')

    parser.add_argument('-dropout', type=float, default=0.2,
                        help='Float between 0 and 1. Fraction of the input units to drop for input gates and '
                             'recurrent connections.')

    parser.add_argument('-valid_per_epoch', type=int, default=25,
                        help='Run validation (if file provided) every n epochs.')

    args = parser.parse_args()
    hyperparameter = vars(args)

    output_files = hyperparameter['output_files'].format(**hyperparameter)
    del hyperparameter['output_files']
    args_file = output_files + '.model_args'
    weights_file = output_files + '.h5py'
    log_file = output_files + '.log'
    hyperparameter.update(dict(args_file=args_file, weights_file=weights_file, log_file=log_file))

    print('')
    print('Hyperparameter')
    for key, value in hyperparameter.items():
        print('{}:\t\t{}'.format(key, value))
    print('')
    print(hyperparameter['valid_file'])
    train(**hyperparameter)


def train(train_file, embeddings_file, args_file, weights_file, log_file, valid_file, lstm_size, neg_sampling,
          max_epochs, patience, batch_size, dropout, valid_per_epoch):
    """
    Start the training workflow. Read the stories train (and validation) file. Load embeddings from given file.
    Create model with given hyperparameter and generate instances for training. Different metrics will be printed while
     training. The best learned weights will be written to a file till the training stops due to given criterions.
    :param train_file: 'Train the model on the stories of this file. Can provides only right ending or additional wrong
    ending. In ".coref" format. See "scripts/coreference"
    :param embeddings_file: Embedding file in word2vec format
    :param args_file: Save model hyperparameter to this file
    :param weights_file: Save learned model weights to this file
    :param log_file: Save training log to this file
    :param valid_file: Validate model during training on this file
    :param lstm_size: Size of the lstm output layer
    :param neg_sampling: Number of negative (wrong) endings sampled (from other right endings) per story
    :param max_epochs: Maximal number of epochs to train for (also valid for early stopping)
    :param patience: Early stopping patience; Number of epochs with no improvement after which training will be stopped
    :param batch_size: Batch size while training
    :param dropout: Float between 0 and 1. Fraction of the input units to drop for input gates and recurrent connections
    :param valid_per_epoch: Run validation (if file provided) every n epochs

    :return: Write trained model to given files. Print training status after every epoch.
    """
    import pickle
    from read_corpus import get_stories
    from utils import read_embeddings
    from keras_utils import BatchStatusCallback, TrainLogger, EarlyStoppingAndSaving
    from instances import get_train_instances, indexify
    from model import get_model
    from validation import ValidationCallback

    # read stories and gather all distinct words in the stories to load less embeddings in the next step
    words = set()
    story_ids_train, contexts_train, right_endings_train, wrong_endings_train = get_stories(train_file, words)
    story_ids_val, contexts_val, right_endings_val, wrong_endings_val = get_stories(valid_file, words)

    # load embeddings (only for those words, occurring in the stories)
    word_to_idx, embeddings, embedding_dim = read_embeddings(embeddings_file, words)
    print('loaded embeddings with shape: ' + str(embeddings.shape))

    # convert words to indices; get maximal input lengths to zero-pad the sequences for keras
    max_input_len = dict(context=0, ending=0)
    indexify(story_ids_train, contexts_train, right_endings_train, wrong_endings_train, word_to_idx, max_input_len)
    indexify(story_ids_val, contexts_val, right_endings_val, wrong_endings_val, word_to_idx, max_input_len)
    max_context_len, max_ending_len = max_input_len['context'], max_input_len['ending']

    # save model hyperparameter
    model_args = dict(word_to_idx=word_to_idx, lstm_size=lstm_size, dropout=dropout, embedding_dim=embedding_dim)
    pickle.dump(model_args, open(args_file, 'w'))
    print('wrote model hyperparameter to {}'.format(args_file))

    # calculate samples per epoch, such that samples_per_epoch % batch_size == 0 to avoid keras warning
    samples_per_epoch = len(story_ids_train) * neg_sampling
    samples_per_epoch += batch_size - (samples_per_epoch % batch_size)

    # initialize callbacks
    train_logger_cb = TrainLogger(log_file)
    # set updates > 0 to see loss during batch, not only at the end of each epoch
    batch_status_update_cb = BatchStatusCallback(samples_per_epoch, batch_size, updates=0)
    # stop if loss isn't decreasing for {patience} epochs by min 0.0
    early_stopping_cb = EarlyStoppingAndSaving(weights_file, monitor='loss', patience=patience,
                                               mode='min', verbose=1, min_delta=0.0)
    callbacks = [batch_status_update_cb, early_stopping_cb, train_logger_cb]
    # only add callback for validation if valid_file is provided
    if valid_file is not None:
        validation_cb = ValidationCallback(story_ids_val, contexts_val, right_endings_val, wrong_endings_val,
                                           max_context_len, max_ending_len, per_epochs=valid_per_epoch,
                                           train_logger=train_logger_cb)
        callbacks.append(validation_cb)

    # generate model and instance generator to train model
    model = get_model(word_to_idx, embedding_dim, max_context_len, max_ending_len, lstm_size, dropout, embeddings)
    instance_generator = get_train_instances(story_ids_train, contexts_train, right_endings_train, wrong_endings_train,
                                             max_context_len, max_ending_len, neg_sampling, batch_size)

    model.fit_generator(instance_generator, samples_per_epoch=samples_per_epoch, nb_epoch=max_epochs, verbose=2,
                        callbacks=callbacks)

if __name__ == '__main__':
    run()

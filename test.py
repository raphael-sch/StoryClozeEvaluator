import argparse
from utils import are_valid_model_files, is_valid_file
import random
random.seed(1337)


def run():
    """
    Read command line arguments and set defaults.
    :return: run the test workflow
    """

    parser = argparse.ArgumentParser(description='Test a trained model for Story Cloze Test')

    parser.add_argument('model_files', type=lambda x: are_valid_model_files(parser, x),
                        help='Basename of the model files. (Without the endings ".model_args" and "h5py")\n'
                             'Example: ./model/size300_neg1')

    parser.add_argument('test_file', type=lambda x: is_valid_file(parser, x),
                        help='Test the saved model on the stories from this file. Must include right and wrong ending.'
                             ' \n In ".coref" format. See "scripts/coreference/"')

    parser.add_argument('-batch_size', nargs='?', type=int, default=128,
                        help='Batch size for prediction run')

    args = parser.parse_args()
    hyperparameter = vars(args)

    args_file, weights_file = hyperparameter['model_files']
    del hyperparameter['model_files']
    hyperparameter.update(dict(args_file=args_file, weights_file=weights_file))

    print('')
    print('Hyperparameter')
    for key, value in hyperparameter.items():
        print('{}:\t\t{}'.format(key, value))
    print('')

    test(**hyperparameter)


def test(args_file, weights_file, test_file, batch_size):
    """
    Read the stories from test file. Load saved model. Run evaluation on the stories from the test file
    :param args_file: "*.model_args" file. Hyperparameter of the saved model
    :param weights_file: ".h5py" file. Weights of the saved model
    :param test_file: story cloze test file to evaluate model on
    :param batch_size: batch size for prediction run
    :return: prints the evaluation metrics for the test file
    """
    import pickle
    from read_corpus import get_stories
    from instances import indexify
    from model import get_model
    from validation import get_validation_data, run_validation

    model_args = pickle.load(open(args_file))
    word_to_idx = model_args['word_to_idx']

    # read stories
    story_ids_test, contexts_test, right_endings_test, wrong_endings_test = get_stories(test_file)

    # convert words to indices; get maximal input lengths to zero-pad the sequences for keras
    max_input_len = dict(context=0, ending=0)
    indexify(story_ids_test, contexts_test, right_endings_test, wrong_endings_test, word_to_idx, max_input_len)
    max_context_len, max_ending_len = max_input_len['context'], max_input_len['ending']
    model_args.update(dict(max_context_len=max_context_len, max_ending_len=max_ending_len))

    # create padded numpy arrays of input data
    validation_data = get_validation_data(story_ids_test, contexts_test, right_endings_test, wrong_endings_test,
                                          max_context_len, max_ending_len)
    # restore model from saved files
    model = get_model(**model_args)
    model.load_weights(weights_file)

    run_validation(model, validation_data, batch_size)


if __name__ == '__main__':
    run()

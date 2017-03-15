import csv
import os


def is_valid_file(parser, filename):
    if not os.path.isfile(filename):
        parser.error("The file %s does not exist!" % filename)
    else:
        return filename


def get_stories(file_name, train_format=True):
    """
    Read the csv file and create dictionaries (story_id as access) for context sentences, right and wrong endings.
    Gather all distinct words occurring in any file.
    :param file_name: name of the csv file of the story cloze test
    :param train_format: whether the file is in train or test format
    :return: list of story ids, id -> context sentences, id -> right ending, id -> wrong ending, distinct words
    """
    stories = dict()
    for line_count, sp in enumerate(csv.reader(open(file_name))):
        if line_count == 0:
            continue
        if line_count == 500:
            continue
        sp = [s.strip() for s in sp]

        # train file differs from test and dev file by having only one (right) ending.
        if train_format:
            idx, sentences, title = sp[0], sp[2:], sp[1]
            context = sentences[:-1]
            right_ending = sentences[-1]
            wrong_ending = ''
        else:
            idx, sentences, label = sp[0], sp[1:7], int(sp[7])
            context, right_ending, wrong_ending = sentences[:4], sentences[4], sentences[5]
            if label == 2:
                right_ending, wrong_ending = wrong_ending, right_ending
        story = context + [right_ending] + [wrong_ending]
        stories[idx] = story

    return stories


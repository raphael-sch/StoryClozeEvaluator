from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

snowball_stemmer = SnowballStemmer('english')
stopwords_english = stopwords.words('english')


def normalize_sentence(sentence, stemming=True, replace_punctuation=True):
    """
    Performs different preprocessing steps on input text. Tokenize input string with 'punkt' tokenizer.
    :param sentence: list of words to be normalized
    :param stemming: do stemming if True
    :param replace_punctuation: replace punctuation with '.' to be fed as punctation embedding
    :return: list of tokens
    """
    if sentence is None:
        return []
    # lowercase all words; remove ','
    text_normalized = [word.lower() for word in sentence if word not in [',']]
    if stemming:
        text_normalized = [snowball_stemmer.stem(word) for word in text_normalized]
    if replace_punctuation:
        text_normalized = ['.' if word in ['.', '?', '!', ':'] else word for word in sentence]
    return text_normalized


def get_stories(file_name, words=None):
    """
    Read the preprocessed file and create dictionaries (story_id as access) for context sentences, right and wrong
    endings. Context are the first 4 sentences, the right ending the 5th and the 6th is the wrong ending if present.
    Gather all distinct words occurring in any file.
    :param file_name: name of the csv file of the story cloze test
    :param words: gather distinct words throughout all stories;
    :return: list of story ids, id -> context sentences, id -> right ending, id -> wrong ending
    """
    story_ids = list()
    contexts = dict()
    right_endings = dict()
    wrong_endings = dict()
    if file_name is None:
        return story_ids, contexts, right_endings, wrong_endings

    print('read stories from filename: ' + str(file_name))
    with open(file_name) as file_reader:
        for line in file_reader:
            line = line.rstrip().split('\t')
            story_id, sentences = line[0], line[1:]
            story_ids.append(story_id)

            sentences = [normalize_sentence(sentence.split(' '), True, True) for sentence in sentences]
            # collect distinct words
            if words is not None:
                [[words.add(word) for word in sentence] for sentence in sentences]

            # put start ending flag at the end of the context input
            context = [word for sentence in sentences[:4] for word in sentence] + ['_START_ENDING_']
            contexts[story_id] = context
            right_ending = sentences[4]
            right_endings[story_id] = right_ending

            # if story has 6 sentences, the the last one is a wrong ending
            if len(sentences) == 6:
                wrong_ending = sentences[5]
                wrong_endings[story_id] = wrong_ending
    print('found {} stories'.format(len(story_ids)))
    return story_ids, contexts, right_endings, wrong_endings


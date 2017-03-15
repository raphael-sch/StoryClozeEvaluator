import argparse
from pycorenlp import StanfordCoreNLP
import nltk
from coref_utils import is_valid_file, get_stories
import os


def run():
    """
    Read command line arguments and set defaults.
    :return: run the coreference resolution workflow
    """

    parser = argparse.ArgumentParser(description='Run the coreference resolution workflow.')

    parser.add_argument('input_file', type=lambda x: is_valid_file(parser, x),
                        help='Input file (csv) in the format provided by the Story Cloze Test')

    parser.add_argument('input_file_format', type=bool,
                        help='True if the input file is in train format (only right ending) or False if in test format'
                             '(right and wrong ending)')

    parser.add_argument('-corenlp_host', type=str,
                        default='http://localhost:9000',
                        help='Hostname of the Stanford CoreNLP Server. See http://stanfordnlp.github.io/CoreNLP/')

    parser.add_argument('-output_folder', type=str,
                        default='./data2/',
                        help='Folder to write the .coref file into. Absolute path preferred')

    args = parser.parse_args()
    arguments = vars(args)

    output_folder = arguments['output_folder']
    if not os.path.exists(output_folder):
        raise ValueError('Can\'t find output folder {}'.format(output_folder))

    output_file = os.path.join(output_folder, os.path.basename(arguments['input_file']) + '.coref')

    write_coreference_file(arguments['corenlp_host'], arguments['input_file'], output_file,
                           arguments['input_file_format'])


def write_coreference_file(corenlp_host, input_file, output_file, train_format, num_prots=2):
    """
    Use Standford CoreNLP to search coreference chains of protagonists. Replace the found tokens with __PROT1__ and
    __PROT2__. Write results to new file.

    :param corenlp_host: Hostname of the Stanford CoreNLP Server. See http://stanfordnlp.github.io/CoreNLP/'
    :param input_file: Input file (csv) in the format provided by the Story Cloze Test'
    :param output_file: Path of the .coref output file
    :param train_format: Weather the input file is in train format (only right ending) or test format
    (right and wrong ending)
    :param num_prots: How many protagonists should be identified and replaced. Model code can only handle 2
    :return: Write .coref file to given location
    """
    print('try to connect to Stanford CoreNLP...{}'.format(corenlp_host))
    nlp = StanfordCoreNLP(corenlp_host)
    print('connected to Stanford CoreNLP')

    print('read stories from: ' + str(input_file))
    # context(0-3), right_ending(4), wrong_ending(5)
    stories = get_stories(input_file, train_format=train_format)

    print('write stories with coreference information to: ' + str(output_file))
    with open(output_file, 'w') as file_writer:
        for i, story_item in enumerate(stories.items()):
            if i % 100 == 0:
                print(str(i) + '/' + str(len(stories)))

            story_id, story = story_item
            # remove non ascii characters
            story = [''.join([c for c in sentence if ord(c) in range(128)]) for sentence in story]
            sentences = [nltk.word_tokenize(sentence) for sentence in story]

            # query Stanford Core NLP server
            output_json = nlp.annotate(' '.join(story), properties={
                'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,mention,coref',
                'coref.algorithm': 'neural',
                'outputFormat': 'json'
            })

            coref_chains = dict()
            # some queries throw 'java.lang.NullPointerException' in CoreNLP; can be identified by this if clause
            if type(output_json) != unicode:
                for coref, value in output_json['corefs'].items():
                    min_id = None
                    coref_chain = dict()
                    for reference in value:
                        # look for references which span only one word are in singular and tagged as animated
                        if reference['animacy'] == 'ANIMATE' and reference['number'] == 'SINGULAR' and \
                                                reference['endIndex'] - reference['startIndex'] == 1:
                            # get minimum coreference id, to get the first two (by mention) coreference chains
                            if min_id is None:
                                min_id = reference['id']
                            min_id = min(min_id, reference['id'])

                            coref_chain[reference['sentNum']-1] = coref_chain.get(reference['sentNum'], [])\
                                                                  + [(reference['startIndex']-1, reference['text'])]
                    if len(coref_chain) > 0:
                        coref_chains[min_id] = coref_chain

            # get first two coreference chains as protagonist one and two
            coref_chains_keys_sorted = sorted(coref_chains.keys())
            for prot_id in range(min(num_prots, len(coref_chains))):
                prot_str = '_PROT{}_'.format(prot_id)
                coref_chain = coref_chains[coref_chains_keys_sorted[prot_id]]
                for sentence_id, word_items in coref_chain.items():
                    for word_id, text in word_items:
                        # original sentences had been concatenate (and independently separated by CoreNLP) in order to
                        # find coreferences. If sentences produced by CoreNLP don't align with original sentences don't
                        # replace coreference
                        try:
                            if sentences[sentence_id][word_id] == text:
                                sentences[sentence_id][word_id] = prot_str
                            else:
                                print(text)
                                print(sentences[sentence_id])
                                print(sentences)
                        except IndexError:
                            pass
            # filter non ascii characters
            sentences = [[''.join([c for c in word if ord(c) in range(128)])
                          for word in sentence] for sentence in sentences]

            # write to output file
            file_writer.write(story_id + '\t' + '\t'.join([' '.join(sentence) for sentence in sentences]))
            file_writer.write('\n')

if __name__ == '__main__':
    run()

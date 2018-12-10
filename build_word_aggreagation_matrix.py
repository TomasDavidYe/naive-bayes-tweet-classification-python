import pandas as pd


def get_word_count_rel_mentions(word):
    return 1


def get_word_count_in_nerel_mentions(word):
    return 2


def get_word_count_in_rel_contexts(word):
    return 3


def get_word_count_in_nerel_contexts(word):
    return 4


def get_word_rank_absolute(word):
    return 5


def get_word_rank_rel(word):
    return 6


def get_word_rank_nerel(word):
    return 7


def check(text, word, relevance):
    if not isinstance(text, str):
        return False
    seq = text.split()
    return word in seq and relevance == 'nerel'


dataset = pd.read_csv('cleaner_data.csv')
mentions = dataset[['Obsah zmínek', 'Štítek']]
contexts = dataset[['Kontext', 'Štítek']]
words = pd.read_csv('word_frequencies.csv')
words.columns = ['word_text', 'absolute_frequency']

top_words = words.iloc[0:20, :]

test = mentions[mentions.apply(lambda x: check(x['Obsah zmínek'], 'ods', x['Štítek']), axis=1)]

test.size


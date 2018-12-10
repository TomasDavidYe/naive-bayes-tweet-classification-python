import pandas as pd

words_in_mentions = pd.read_csv('counts_in_mentions.csv')
words_in_mentions.columns = ['word_text', 'absolute_frequency']

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



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


def check_if_word_is_in_text(text, word):
    if not isinstance(text, str):
        return False
    seq = text.split()
    return word in seq


def check_if_word_is_in_relevant_text(text, word, relevance):
    return check_if_word_is_in_text(text, word) and relevance == 'rel'


def check_if_word_is_in_irrelevant_text(text, word, relevance):
    return check_if_word_is_in_text(text, word) and relevance == 'nerel'


dataset = pd.read_csv('cleaner_data.csv')
mentions_with_relevance = dataset[['Obsah zmínek', 'Štítek']].dropna()
words_in_mentions = pd.read_csv('counts_in_mentions.csv')
words_in_mentions.columns = ['word_text', 'absolute_frequency']

selection1 = mentions_with_relevance.apply(func=lambda x: check_if_word_is_in_text(word='usa', text=x['Obsah zmínek']) ,axis=1)
test1 = mentions_with_relevance.loc[selection1, :]
len(test1)
selection2 = mentions_with_relevance.apply(func=lambda x: check_if_word_is_in_relevant_text(word='usa',
                                                                                            text=x['Obsah zmínek'],
                                                                                            relevance=x['Štítek']),
                                           axis=1)
test2 = mentions_with_relevance.loc[selection2, :]

selection3 = mentions_with_relevance.apply(func=lambda x: check_if_word_is_in_irrelevant_text(word='usa',
                                                                                            text=x['Obsah zmínek'],
                                                                                            relevance=x['Štítek']),
                                           axis=1)
test3 = mentions_with_relevance.loc[selection3, :]
len(test1)
len(test2)
len(test3)


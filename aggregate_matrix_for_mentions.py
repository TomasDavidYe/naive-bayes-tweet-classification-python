import pandas as pd
import numpy as np


abs_count = pd.read_csv('resources/absolute_occurrence_count.csv')
count = pd.read_csv('resources/occurrences_in_all_mentions_count.csv')
rel_count = pd.read_csv('resources/occurrences_in_rel_mentions_count.csv')
nerel_count = pd.read_csv('resources/occurrences_in_nerel_mentions_count.csv')

abs_count.index[abs_count['word_text'] == 'hovno'][0]
ranked_abs = abs_count.rank(numeric_only=True, ascending=False)
ranked = count.rank(numeric_only=True, ascending=False)
ranked_rel = rel_count.rank(numeric_only=True, ascending=False)
ranked_nerel = nerel_count.rank(numeric_only=True, ascending=False)


def get_rank(word):
    i = count.index[count['word_text'] == word][0]
    return int(ranked.iloc[i]['count'])


def get_rank_rel(word):
    i = rel_count.index[rel_count['word_text'] == word][0]
    return int(ranked_rel.iloc[i]['count'])


def get_rank_nerel(word):
    i = nerel_count.index[nerel_count['word_text'] == word][0]
    return int(ranked_nerel.iloc[i]['count'])


def get_rel_count(word):
    i = rel_count.index[rel_count['word_text'] == word][0]
    return rel_count.iloc[i]['count']


def get_nerel_count(word):
    i = nerel_count.index[nerel_count['word_text'] == word][0]
    return nerel_count.iloc[i]['count']





columns = [
    'word_text',
    'word_count_zminka',
    'word_count_zminka_rel',
    'word_count_zminka_nerel',
    'word_zminka_rank',
    'word_zminka_rank_rel',
    'word_zminka_rank_nerel'
]

N = 500;
aggregation_matrix_words = pd.DataFrame(data=[], columns=columns)
top_words = count.loc[:N, :]
for i, word_field in enumerate(top_words.values):
    print(i)
    word = list(word_field)[0]
    value = list(word_field)[1]
    row = pd.Series(index=columns,
                    data=[
                        word, value, get_rel_count(word), get_nerel_count(word), get_rank(word), get_rank_rel(word), get_rank_nerel(word)
                    ])
    aggregation_matrix_words = aggregation_matrix_words.append(row, ignore_index=True)


aggregation_matrix_words.to_csv('matrices/aggregation_matrix_words.csv')








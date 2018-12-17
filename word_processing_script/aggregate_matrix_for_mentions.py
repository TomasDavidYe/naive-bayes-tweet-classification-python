import pandas as pd
import datetime
from helper_methods import get_columns_for_word_aggregate_matrix


occurrences_count_absolute = pd.read_csv('../resources/mentions/latest/occurrences_count_absolute.csv', names=['word_text', 'count'])
occurrences_count_all = pd.read_csv('../resources/mentions/latest/occurrences_count_all.csv', names=['word_text', 'count'])
occurrences_count_rel = pd.read_csv('../resources/mentions/latest/occurrences_count_rel.csv', names=['word_text', 'count'])
occurrences_count_nerel = pd.read_csv('../resources/mentions/latest/occurrences_count_absolute.csv', names=['word_text', 'count'])


ranked_abs = occurrences_count_absolute.rank(numeric_only=True, ascending=False)
ranked_all = occurrences_count_all.rank(numeric_only=True, ascending=False)
ranked_rel = occurrences_count_rel.rank(numeric_only=True, ascending=False)
ranked_nerel = occurrences_count_nerel.rank(numeric_only=True, ascending=False)


def get_rank(word):
    i = occurrences_count_all.index[occurrences_count_all['word_text'] == word][0]
    return int(ranked_all.iloc[i]['count'])


def get_rank_rel(word):
    i = occurrences_count_rel.index[occurrences_count_rel['word_text'] == word][0]
    return int(ranked_rel.iloc[i]['count'])


def get_rank_nerel(word):
    i = occurrences_count_nerel.index[occurrences_count_nerel['word_text'] == word][0]
    return int(ranked_nerel.iloc[i]['count'])


def get_rel_count(word):
    i = occurrences_count_rel.index[occurrences_count_rel['word_text'] == word][0]
    return occurrences_count_rel.iloc[i]['count']


def get_nerel_count(word):
    i = occurrences_count_nerel.index[occurrences_count_nerel['word_text'] == word][0]
    return occurrences_count_nerel.iloc[i]['count']


def create_matrix_with_top_n_words(n):
    columns = get_columns_for_word_aggregate_matrix()
    aggregation_matrix_words = pd.DataFrame(data=[], columns=columns)
    top_words = occurrences_count_all.loc[:n, :]
    for i, word_field in enumerate(top_words.values):
        print(i)
        word = list(word_field)[0]
        value = list(word_field)[1]
        row = pd.Series(index=columns,
                        data=[
                            word, value, get_rel_count(word), get_nerel_count(word), get_rank(word), get_rank_rel(word),
                            get_rank_nerel(word)
                        ])
        aggregation_matrix_words = aggregation_matrix_words.append(row, ignore_index=True)
    time = datetime.datetime.now().strftime('%c').replace(' ', '_')
    aggregation_matrix_words.to_csv('../resources/aggregation_matrices/words/aggregation_matrix_words_' + time + '.csv')


create_matrix_with_top_n_words(20)
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from general_data_scripts.stop_words_script import get_stop_words_new

print('Creating stop words list')
stop_words_cz = get_stop_words_new()


def create_feature_matrix(matrix, tokens) -> pd.DataFrame:
    doc_names = ['mention_{:d}'.format(i) for i, value in enumerate(matrix)]
    return pd.DataFrame(data=matrix, index=doc_names, columns=tokens).transpose()


def get_indicator_matrix(feature_matrix):
    return feature_matrix.apply(lambda x: x > 0).astype(dtype=int)


def get_relevance_indicator_matrix(indicator_matrix):
    return indicator_matrix.loc[:, indicator_matrix.apply(func=lambda x: x[RELEVANT] == 1, axis=0)]


def get_irrelevance_indicator_matrix(indicator_matrix):
    return indicator_matrix.loc[:, indicator_matrix.apply(func=lambda x: x[RELEVANT] == 0, axis=0)]


RELEVANT = 'zzzrelevantzzz'


def tag_with_relevance(mention, relevance):
    if relevance == 'rel':
        return mention + ' ' + RELEVANT
    else:
        return mention


def vectorize():
    print('Loading...')
    dataset = pd.read_csv('../resources/general_data/cleaner_data.csv')
    mentions = dataset[['Obsah zmínek', 'Štítek']].dropna().apply(func=lambda x: tag_with_relevance(x['Obsah zmínek'], x['Štítek']), axis=1)
    vectorizer = CountVectorizer(stop_words=stop_words_cz)
    print('Fitting...')
    matrix_raw = vectorizer.fit_transform(mentions).todense()
    print('Making matrices...')
    feature_names = vectorizer.get_feature_names()
    feature_matrix = create_feature_matrix(matrix_raw, feature_names)
    indicator_matrix = get_indicator_matrix(feature_matrix)
    indicator_matrix_for_relevance = get_relevance_indicator_matrix(indicator_matrix)
    indicator_matrix_for_irrelevance = get_irrelevance_indicator_matrix(indicator_matrix)
    feature_matrix.drop(index=[RELEVANT], inplace=True)
    indicator_matrix.drop(index=[RELEVANT], inplace=True)
    indicator_matrix_for_irrelevance.drop(index=[RELEVANT], inplace=True)
    indicator_matrix_for_relevance.drop(index=[RELEVANT], inplace=True)

    print('Summing counts...')
    # TODO optimize with matrix multiplication
    absolute_occurrence_count = feature_matrix.apply(func=np.sum, axis=1)
    occurrences_in_all_mentions_count = indicator_matrix.apply(func=np.sum, axis=1)
    occurrences_in_rel_mentions_count = indicator_matrix_for_relevance.apply(func=np.sum, axis=1)
    occrrences_in_nerel_mentions_count = indicator_matrix_for_irrelevance.apply(func=np.sum, axis=1)
    print('Sorting...')
    absolute_occurrence_count.sort_values(inplace=True, ascending=False)
    occurrences_in_all_mentions_count.sort_values(inplace=True, ascending=False)
    occurrences_in_rel_mentions_count.sort_values(inplace=True, ascending=False)
    occrrences_in_nerel_mentions_count.sort_values(inplace=True, ascending=False)

    print('Saving...')
    absolute_occurrence_count.to_csv('../resources/mentions/absolute_occurrence_count.csv')
    occurrences_in_all_mentions_count.to_csv('../resources/mentions/occurrences_in_all_mentions_count.csv')
    occurrences_in_rel_mentions_count.to_csv('../resources/mentions/occurrences_in_rel_mentions_count.csv')
    occrrences_in_nerel_mentions_count.to_csv('../resources/mentions/occurrences_in_nerel_mentions_count.csv')

vectorize()

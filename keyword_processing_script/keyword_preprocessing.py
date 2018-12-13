import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


path_to_data = 'resources/general_data/cleaner_data.csv'
path_to_data = '../' + path_to_data
dataset = pd.read_csv(path_to_data)[['Klíčová slova', 'Štítek']]


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


keywords = dataset.dropna().apply(func=lambda x: tag_with_relevance(x['Klíčová slova'], x['Štítek']), axis=1)
vectorizer = CountVectorizer()
matrix_raw = vectorizer.fit_transform(keywords).todense()
feature_names = vectorizer.get_feature_names()


feature_matrix = create_feature_matrix(matrix_raw, feature_names)
indicator_matrix = get_indicator_matrix(feature_matrix)
indicator_matrix_for_relevance = get_relevance_indicator_matrix(indicator_matrix)
indicator_matrix_for_irrelevance = get_irrelevance_indicator_matrix(indicator_matrix)


feature_matrix.drop(index=[RELEVANT], inplace=True)
indicator_matrix.drop(index=[RELEVANT], inplace=True)
indicator_matrix_for_irrelevance.drop(index=[RELEVANT], inplace=True)
indicator_matrix_for_relevance.drop(index=[RELEVANT], inplace=True)


absolute_occurrence_count = feature_matrix.apply(func=np.sum, axis=1)
occurrences_in_all_mentions_count = indicator_matrix.apply(func=np.sum, axis=1)
occurrences_in_rel_mentions_count = indicator_matrix_for_relevance.apply(func=np.sum, axis=1)
occrrences_in_nerel_mentions_count = indicator_matrix_for_irrelevance.apply(func=np.sum, axis=1)


absolute_occurrence_count.sort_values(inplace=True, ascending=False)
occurrences_in_all_mentions_count.sort_values(inplace=True, ascending=False)
occurrences_in_rel_mentions_count.sort_values(inplace=True, ascending=False)
occrrences_in_nerel_mentions_count.sort_values(inplace=True, ascending=False)


temp = []
keywords = occurrences_in_all_mentions_count.index
for keyword in keywords:
    row = {}
    row['keyword'] = keyword
    row['keyword_count'] = occurrences_in_all_mentions_count[keyword]
    row['keyword_count_rel'] = occurrences_in_rel_mentions_count[keyword]
    row['keyword_count_nerel'] = occrrences_in_nerel_mentions_count[keyword]
    temp.append(row)


aggregate_matrix_keywords = pd.DataFrame(data=temp)[['keyword', 'keyword_count', 'keyword_count_rel', 'keyword_count_nerel']]
aggregate_matrix_keywords.to_csv('../resources/matrices/aggregate_matrix_keywords_13_12_2018.csv')







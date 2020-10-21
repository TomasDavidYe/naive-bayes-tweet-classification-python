import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime

RELEVANT = 'zzzrelevantzzz'


def tag_with_relevance(mention, relevance):
    if relevance == 'rel':
        return mention + ' ' + RELEVANT
    else:
        return mention


def create_feature_matrix(matrix, tokens) -> pd.DataFrame:
    doc_names = ['mention_{:d}'.format(i) for i, value in enumerate(matrix)]
    return pd.DataFrame(data=matrix, index=doc_names, columns=tokens).transpose()


def get_indicator_matrix(feature_matrix):
    return feature_matrix.apply(lambda x: x > 0).astype(dtype=int)


def get_relevance_indicator_matrix(indicator_matrix):
    return indicator_matrix.loc[:, indicator_matrix.apply(func=lambda x: x[RELEVANT] == 1, axis=0)]


def get_irrelevance_indicator_matrix(indicator_matrix):
    return indicator_matrix.loc[:, indicator_matrix.apply(func=lambda x: x[RELEVANT] == 0, axis=0)]


def get_vectorization_for_keywords(ratio):
    dataset = pd.read_csv('resources/source_data/cleaner_data.csv')[['Klíčová slova', 'Štítek']]
    train_set_size = int(len(dataset) * ratio)
    dataset = dataset[:train_set_size]
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

    occurrences_count_absolute = feature_matrix.apply(func=np.sum, axis=1)
    occurrences_count_all = indicator_matrix.apply(func=np.sum, axis=1)
    occurrences_count_rel = indicator_matrix_for_relevance.apply(func=np.sum, axis=1)
    occurrences_count_nerel = indicator_matrix_for_irrelevance.apply(func=np.sum, axis=1)

    occurrences_count_absolute.sort_values(inplace=True, ascending=False)
    occurrences_count_all.sort_values(inplace=True, ascending=False)
    occurrences_count_rel.sort_values(inplace=True, ascending=False)
    occurrences_count_nerel.sort_values(inplace=True, ascending=False)

    return [occurrences_count_all, occurrences_count_rel, occurrences_count_nerel]


def create_aggregation_matrix_for_keywords(ratio=1.0):
    [occurrences_count_all, occurrences_count_rel, occurrences_count_nerel] = get_vectorization_for_keywords(ratio)
    temp = []
    keywords = occurrences_count_all.index
    for keyword in keywords:
        row = {}
        row['keyword'] = keyword
        row['keyword_count'] = occurrences_count_all[keyword]
        row['keyword_count_rel'] = occurrences_count_rel[keyword]
        row['keyword_count_nerel'] = occurrences_count_nerel[keyword]
        temp.append(row)

    aggregate_matrix_keywords = pd.DataFrame(data=temp)[
        ['keyword', 'keyword_count', 'keyword_count_rel', 'keyword_count_nerel']]
    time = datetime.now().strftime('%c').replace(' ', '_')
    aggregate_matrix_keywords.to_csv('resources/aggregation_matrices/keywords/aggregation_matrix_keywords_' + time + '.csv')


create_aggregation_matrix_for_keywords(0.8)







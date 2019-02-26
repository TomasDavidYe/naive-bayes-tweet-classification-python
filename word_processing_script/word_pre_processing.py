import pandas as pd
import numpy as np
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from helper_methods import get_stop_words
import os


def rename_columns_for_feature_matrix(matrix, tokens) -> pd.DataFrame:
    doc_names = ['mention_{:d}'.format(i + 2) for i, value in enumerate(matrix)]
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


def get_data_for_vectorization(month, ratio):
    print('Loading...')
    source_path = '../resources/source_data/cleaner_data_' + month + '.csv'
    dataset = pd.read_csv(source_path).set_index('id')
    dataset.sort_values(by='Datum vytvoření', inplace=True)
    train_length = int(len(dataset) * ratio)
    temp = dataset[['Obsah zmínek', 'Štítek']].dropna().apply(func=lambda x: tag_with_relevance(x['Obsah zmínek'], x['Štítek']), axis=1)
    train_set = temp.iloc[:train_length]
    test_set = temp.iloc[train_length:]
    return [train_set, test_set]


def get_feature_matrix(month, ratio):
    [mentions_train, mentions_test] = get_data_for_vectorization(month ,ratio)
    print('Creating stop words list')
    stop_words_cz = get_stop_words()
    vectorizer = CountVectorizer(stop_words=stop_words_cz)
    print('Fitting...')
    matrix_raw = vectorizer.fit_transform(mentions_train).todense()
    print('Making matrices...')
    feature_names = vectorizer.get_feature_names()
    return rename_columns_for_feature_matrix(matrix_raw, feature_names)


def vectorize(month, ratio=1.0):
    feature_matrix = get_feature_matrix(month, ratio)
    indicator_matrix = get_indicator_matrix(feature_matrix)
    indicator_matrix_for_relevance = get_relevance_indicator_matrix(indicator_matrix)
    indicator_matrix_for_irrelevance = get_irrelevance_indicator_matrix(indicator_matrix)

    print('Dropping unnecessary row')
    feature_matrix.drop(index=[RELEVANT], inplace=True)
    indicator_matrix.drop(index=[RELEVANT], inplace=True)
    indicator_matrix_for_irrelevance.drop(index=[RELEVANT], inplace=True)
    indicator_matrix_for_relevance.drop(index=[RELEVANT], inplace=True)

    print('Summing counts...')
    # TODO optimize with matrix multiplication
    occurrences_count_absolute = feature_matrix.apply(func=np.sum, axis=1)
    occurrences_count_all = indicator_matrix.apply(func=np.sum, axis=1)
    occurrences_count_rel = indicator_matrix_for_relevance.apply(func=np.sum, axis=1)
    occurrences_count_nerel = indicator_matrix_for_irrelevance.apply(func=np.sum, axis=1)

    print('Sorting...')
    occurrences_count_absolute.sort_values(inplace=True, ascending=False)
    occurrences_count_all.sort_values(inplace=True, ascending=False)
    occurrences_count_rel.sort_values(inplace=True, ascending=False)
    occurrences_count_nerel.sort_values(inplace=True, ascending=False)

    print('Naming')
    occurrences_count_absolute.rename('count')
    occurrences_count_all.rename('count')
    occurrences_count_rel.rename('count')
    occurrences_count_nerel.rename('count')


    print('Archiving')
    archive_matrix(occurrences_count_absolute, month, 'occurrences_count_absolute', ratio)
    archive_matrix(occurrences_count_absolute, month, 'occurrences_count_all', ratio)
    archive_matrix(occurrences_count_absolute, month, 'occurrences_count_rel', ratio)
    archive_matrix(occurrences_count_absolute, month, 'occurrences_count_nerel', ratio)

    print('Updating latest')
    dir_name = '../resources/word_vectorization_matrices/' + month + '/latest/occurrences_count_'
    occurrences_count_absolute.to_csv(dir_name + 'absolute.csv')
    occurrences_count_all.to_csv(dir_name + 'all.csv')
    occurrences_count_rel.to_csv(dir_name + 'rel.csv')
    occurrences_count_nerel.to_csv(dir_name + 'nerel.csv')


def archive_matrix(matrix, month, file_name, ratio):
    time = datetime.datetime.now().strftime('%c').replace(' ', '_')
    dir_name = '../resources/word_vectorization_matrices/' + month + '/' + time
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    suffix = '_ratio_' + str(ratio) + '.csv'
    path = os.path.join(dir_name, file_name + suffix)
    matrix.to_csv(path)


# vectorize('prosinec', 0.7)
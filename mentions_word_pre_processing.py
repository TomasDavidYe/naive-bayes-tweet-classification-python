import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from load_data import clear_diacritics_from
from stop_words import get_stop_words

stop_words_cz = get_stop_words('czech')
for index, value in enumerate(stop_words_cz):
    stop_words_cz[index] = clear_diacritics_from(value)
stop_words_cz += [
    'by',
    'take',
    'si',
    'pokud',
    'tomto',
    'az',
    'nej',
    'dalsi',
    'aby',
    'byt',
    'mel',
    'sve',
    'tom',
    'ani',
    'ho',
    'ji',
    'tomu',
    'zatim',
    'jako',
    'toho',
    'tim',
    'tu',
    'pri',
    'proto',
    'bych',
    'mu',
    'pod',
    'coz',
    'jiz',
    'tech'
]


def create_feature_matrix(matrix, tokens) -> pd.DataFrame:
    doc_names = ['mention_{:d}'.format(i) for i, value in enumerate(matrix)]
    return pd.DataFrame(data=matrix, index=doc_names, columns=tokens).transpose()


# TODO implement these methods so that the feature matrix can be created more easily
def get_indicator_matrix(feature_matrix):
    return feature_matrix.apply(lambda x: x > 0).astype(dtype=int)


def get_relevance_indicator_matrix(indicator_matrix):
    return indicator_matrix.loc[:, indicator_matrix.apply(func=lambda x: x[RELEVANT] == 1, axis=0)]


def get_irrelevance_indicator_matrix(indicator_matrix):
    return indicator_matrix.loc[:, indicator_matrix.apply(func=lambda x: x[RELEVANT] == 0, axis=0)]


def run_vectorizer_without_preprocessing():
    dataset = pd.read_csv('cleaner_data.csv')
    mentions = dataset['Obsah zmínek'].dropna()
    vectorizer = CountVectorizer(stop_words=stop_words_cz)
    matrix_raw = vectorizer.fit_transform(mentions).todense()
    feature_names = vectorizer.get_feature_names()
    feature_matrix = create_feature_matrix(matrix_raw, feature_names)


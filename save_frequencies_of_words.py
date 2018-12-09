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
    'zatim'

]

def create_feature_matrix(matrix, tokens) -> pd.DataFrame:
        doc_names = ['Doc {:d}'.format(i) for i, value in enumerate(matrix)]
        return pd.DataFrame(data=matrix, index=doc_names, columns=tokens).transpose()


dataset = pd.read_csv('cleaner_data.csv')
mentions = dataset['Obsah zmínek']
contexts = dataset['Kontext']
text_data_frame = mentions.map(str) + " " + contexts.map(str)
vectorizer = CountVectorizer(stop_words=stop_words_cz)
matrix_raw = vectorizer.fit_transform(text_data_frame).todense()
feature_names = vectorizer.get_feature_names()
feature_matrix = create_feature_matrix(matrix_raw, feature_names)
num_of_columns = len(list(feature_matrix.columns.values))
ones = pd.DataFrame(data=[1 for i in range(0, num_of_columns)], index=feature_matrix.columns, columns=['Frequency'])
counts = feature_matrix.dot(ones)
counts = counts.sort_values(by=['Frequency'], ascending=False)
counts.to_csv('word_frequencies.csv')
print('Hotovo')

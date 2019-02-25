import pandas as pd
import numpy as np
from word_processing_script.word_pre_processing import get_feature_matrix

occurrences_count_all = pd.read_csv('../resources/word_vectorization_matrices/latest/occurrences_count_all.csv', names=['word', 'count'])
occurrences_count_all = occurrences_count_all.loc[occurrences_count_all['count'] >= 42]



def get_segments(data):
    length_of_segment = int(len(data) / 20)
    segments = {}
    for i in range(0, 19):
        segments[100 - 5 * i] = list(data[length_of_segment * i: length_of_segment * (i + 1)]['word'])
    segments[5] = list(data[19*length_of_segment:])
    return segments


segments = get_segments(occurrences_count_all)
feature_matrix = get_feature_matrix(1)
top_words_matrix = feature_matrix.loc[segments[100], :]
sums = top_words_matrix.apply(np.sum)

id_tag = 'id'
zminka_tag = 'Obsah zmínek'
from sklearn.feature_extraction.text import CountVectorizer
data = pd.read_csv('../resources/source_data/cleaner_data.csv').dropna(subset=['Obsah zmínek']).iloc[:10, [1, 3]]
zminky = data[zminka_tag]

vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(zminky).todense()
tokens = vectorizer.get_feature_names()


def get_matrix_with_descriptions(M, tokens):
    rows = data['id']
    columns = tokens
    return pd.DataFrame(index=rows, columns=columns, data=M)


feature_matrix = get_matrix_with_descriptions(matrix, tokens)


index = 2
row = feature_matrix.iloc[index]
id = row.name
words = list(row.loc[row > 0].index)
zminka = data.loc[data['id'] == id][zminka_tag].iloc[0]

for word in words:
    print(word in zminka.split(' '))

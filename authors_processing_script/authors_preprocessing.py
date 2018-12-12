import pandas as pd
import numpy as np
from helper_methods import *

column_names_for_dropping = get_column_names_for_dropping()
dataset = pd.read_csv('resources/general_data/cleaner_data.csv').dropna(subset=['Autor']).drop(columns=column_names_for_dropping).fillna(0)
authors_column = dataset['Autor']
value_counts = authors_column.value_counts()
authors = value_counts.keys()


def get_features_for_author(author):
    result = {}
    author_data = dataset.loc[dataset['Autor'] == author].copy()
    result['author'] = author
    result['author_count_total'] = len(author_data)
    result['author_count_rel'] = len(author_data.loc[author_data['Štítek'] == 'rel'])
    result['author_count_nerel'] = len(author_data.loc[author_data['Štítek'] == 'nerel'])
    author_data.drop(columns=['Autor', 'Štítek'], inplace=True)
    averages = author_data.apply(np.average)
    for column_name in author_data.columns.values:
        result['author_avg_' + column_name] = averages[column_name]
    return result



temp = []
for author in authors:
    temp.append(get_features_for_author(author))
aggregate_matrix_authors = pd.DataFrame(temp)

temp = aggregate_matrix_authors.columns.tolist()[1:-3]
temp.remove('author_avg_followers')
temp.remove('author_avg_fans')
temp.remove('author_avg_favs')
temp.remove('author_avg_influenceScore')
cols = ['author',
        'author_count_total',
        'author_count_rel',
        'author_count_nerel',
        'author_avg_influenceScore',
        'author_avg_followers',
        'author_avg_fans',
        'author_avg_favs'] + temp
aggregate_matrix_authors = aggregate_matrix_authors[cols]
aggregate_matrix_authors.to_csv('resources/matrices/aggregations_matrix_authors_12_12_2018.csv')




A = [1, 2, 3, 4, 5]
A[-1:]
A[:-1]
A[:1]
A[3:]


B = [1, 2, 3, 4, 5, 6, 7, 8, 9]
T = [1]
C = B[1:]
D = C[-1:] + C[:-1]
E = D[-1:] + D[1:-1]





A = {
    'Tomas': 11,
    'Hanka': 22,
    'Tommy': 33,
}

B = pd.Series(data=A)

C = pd.DataFrame(columns=['Tomas', 'Hanka', 'Prdel'], data=[[1, 2, 3],
                                                            [4, 5, 6],
                                                            [7, 8, 9]])
C.append(B, ignore_index=True)


D = [
    {'C1': 1, 'C2': 2, 'C3': 3},
    {'C1': 4, 'C2': 5, 'C3': 6},
    {'C1': 7, 'C2': 8, 'C3': 9}
]
E = pd.DataFrame(data=D)

G = {}
G['Tommy'] = 11
H = column_names_for_aggregate_matrix.copy()
H.remove('author')
H
column_names_for_aggregate_matrix


A = [1, 2, 3, 4]
B = {1, 2, 3, 4}

C = set(A)
C.remove(1)
D = A
D.remove(1)
E = list(A)
E.remove(1)

F = A.copy()
F.remove(1)

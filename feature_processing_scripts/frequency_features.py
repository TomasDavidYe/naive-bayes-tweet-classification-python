import pandas as pd

dataset = pd.read_csv('resources/general_data/cleaner_data.csv')
occurrences_count_all = pd.read_csv('resources/word_vectorization_matrices/latest/occurrences_count_all.csv', names=['word', 'count'])
occurrences_count_rel = pd.read_csv('resources/word_vectorization_matrices/latest/occurrences_count_rel.csv', names=['word', 'count'])
occurrences_count_nerel = pd.read_csv('resources/word_vectorization_matrices/latest/occurrences_count_nerel.csv', names=['word', 'count'])


occurrences_count_all.rank



A = pd.DataFrame(index=['R1', 'R2', 'R3'], columns=['C1', 'C2', 'C3'], data=[[1, 2, 3],
                                                                             [4, 5, 6],
                                                                             [7, 8, 9]])


B = A.loc[['R1', 'R3'] ,['C1', 'C2']]
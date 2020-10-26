import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np

from helper_methods import get_stop_words

full_dataset = pd.read_csv('./resources/source_data/cleaner_data_rijen_prosinec.csv')
data = full_dataset[['id']]
data['text'] = full_dataset['Obsah zmínek']
data['is_relevant'] = full_dataset['Štítek'].map(lambda x: int(x == 'rel'))
data = data.drop_duplicates(subset=['id'])


def get_relevant_words():
    return list(
        pd.read_csv('./resources/word_vectorization_matrices/rijen_prosinec/latest/occurrences_count_absolute.csv')[
            'Unnamed: 0'][:20])


def rename_columns_for_feature_matrix_with_index(matrix, tokens, index):
    relevant_words = list(set(get_relevant_words()).intersection(set(tokens)))
    return pd.DataFrame(data=matrix, index=index, columns=tokens)[relevant_words], relevant_words


def get_pctg_of_relevant_class(s):
    all_samples_count = len(s)
    relevant_samples_count = len(s[s == 1])
    irrelevant_samples_count = len(s[s == 0])
    pctg_rel = 100 * relevant_samples_count / all_samples_count
    return pctg_rel


X = data['text']
y = data['is_relevant']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,
                                                    test_size=0.25)

pctg_rel_total = get_pctg_of_relevant_class(data['is_relevant'])
pctg_rel_train = get_pctg_of_relevant_class(y_train)
pctg_rel_test = get_pctg_of_relevant_class(y_test)

num_of_rows = 100

corpus = X_train[:num_of_rows]
y = y_train[:num_of_rows]
vectorizer = CountVectorizer(stop_words=get_stop_words())
vectorizer.fit(raw_documents=corpus)
len(vectorizer.get_feature_names())

X_matrix_raw = vectorizer.transform(corpus).todense()
X_matrix, words = rename_columns_for_feature_matrix_with_index(matrix=X_matrix_raw,
                                                               tokens=vectorizer.get_feature_names(),
                                                               index=corpus.index)

X_matrix['text'] = corpus
X_matrix['is_relevant'] = y

len(X_matrix[X_matrix['is_relevant'] == 1])

clf = MultinomialNB()
clf.fit(X=X_matrix[words], y=y)

X_matrix['is_relevant_predicted'] = clf.predict(X_matrix[words])
relevance_prob = clf.predict_proba(X_matrix[words])[:, 1]

X_matrix['is_relevant_predicted_proba'] = relevance_prob

feature_prob = pd.DataFrame(data=clf.feature_log_prob_, columns=words).apply(np.exp)
feature_count = pd.DataFrame(data=clf.feature_count_, columns=words)
prior_counts = clf.class_count_
prior_log_prob = clf.class_log_prior_


def calculate_test_probability(test_case_index):
    test_case = X_matrix.loc[[test_case_index], :]
    temp = X_matrix.loc[test_case_index, words]
    computation = pd.DataFrame(index=words)
    p_x_y_1 = feature_prob.iloc[1, :]
    computation['counts'] = temp
    computation['prob_class_0'] = feature_prob.iloc[0, :]
    computation['prob_class_1'] = feature_prob.iloc[1, :]
    computation = computation[['counts', 'prob_class_0', 'prob_class_1']]

    computation['p_x_y_0'] = computation.index.map(
        lambda i: computation.loc[i, 'prob_class_0'] if computation.loc[i, 'counts'] >= 1 else 1 - computation.loc[
            i, 'prob_class_0'])
    computation['p_x_y_1'] = computation.index.map(
        lambda i: computation.loc[i, 'prob_class_1'] if computation.loc[i, 'counts'] >= 1 else 1 - computation.loc[
            i, 'prob_class_1'])

    prior_0 = len(X_matrix[X_matrix['is_relevant'] == 0]) / num_of_rows
    prior_1 = len(X_matrix[X_matrix['is_relevant'] == 1]) / num_of_rows

    p_x_and_class_0 = computation['p_x_y_0'].product() * prior_0
    p_x_and_class_1 = computation['p_x_y_1'].product() * prior_1

    p_class_0_x = p_x_and_class_0 / (p_x_and_class_0 + p_x_and_class_1)
    p_class_1_x = p_x_and_class_1 / (p_x_and_class_0 + p_x_and_class_1)

    print(f'P(Y = 1 | X) = {p_class_1_x}')


calculate_test_probability(9923)

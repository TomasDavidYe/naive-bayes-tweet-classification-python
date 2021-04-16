import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
from matplotlib import pyplot as plt

from constants import WORKING_DIRECTORY, TEXT, RELEVANT
from helper_methods import get_stop_words


def run_optimisation(data, num_of_folds=3):
    X = data[TEXT]
    y = data[RELEVANT]
    print(f'Percentage of relevant in FULL SET = {get_pctg_of_relevant_class(data[RELEVANT])}')
    skf = StratifiedKFold(n_splits=num_of_folds)
    fold_number = 0
    for train_index, test_index in skf.split(X, y):
        fold_number += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        run_single_fold(X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                        fold_number=fold_number)





def run_single_fold(X_train, X_test, y_train, y_test, fold_number):
    print(f'-----------------------FOLD {fold_number} START-----------------------------------')
    print(f'Train set size = {len(X_train)}')
    print(f'Test set size = {len(X_test)}')
    label_train = f'TRAIN_FOLD_{fold_number}'
    label_test = f'TEST_FOLD_{fold_number}'
    print(f'Percentage of relevant in {label_train} SET = {get_pctg_of_relevant_class(y_train)}')
    print(f'Percentage of relevant in {label_test} SET = {get_pctg_of_relevant_class(y_test)}')

    vectorizer = fit_vectorizer(corpus=X_train)
    relevant_words = get_relevant_words(vectorizer=vectorizer, num_of_relevant_words=1000)

    feature_matrix_train = transform(vectorizer=vectorizer,
                                     data=X_train,
                                     relevant_words=relevant_words,
                                     label=label_train)

    feature_matrix_test = transform(vectorizer=vectorizer,
                                    data=X_test,
                                    relevant_words=relevant_words,
                                    label=label_test)

    classifier = train(feature_matrix_train, y_train)

    train_predictions, train_probabilities = predict(classifier, feature_matrix_train, label=label_train)
    test_predictions, test_probabilities = predict(classifier, feature_matrix_test, label=label_test)

    analyze_performance(ground_truth=y_train,
                        predictions=train_predictions,
                        probabilities=train_probabilities,
                        label=label_train)

    analyze_performance(ground_truth=y_test,
                        predictions=test_predictions,
                        probabilities=test_probabilities,
                        label=label_test)

    print(f'-----------------------FOLD {fold_number} END-------------------------------------')


def get_relevant_words(vectorizer, num_of_relevant_words=50):
    print('Constructing relevant vocabulary...')
    return list(set(get_n_most_used_words(num_of_relevant_words)).intersection(set(vectorizer.get_feature_names())))


def fit_vectorizer(corpus):
    print('Fitting vectorizer...')
    vectorizer = CountVectorizer(stop_words=get_stop_words())
    vectorizer.fit(raw_documents=corpus)
    return vectorizer


def train(features, target):
    print(f'Training model...')
    classifier = MultinomialNB()
    classifier.fit(X=features, y=target)

    return classifier


def transform(vectorizer, data, relevant_words, label=''):
    print(f'Crating feature matrix for {label} SET...')
    feature_matrix_raw = vectorizer.transform(data).todense()
    feature_matrix = rename_columns_for_feature_matrix_with_index(matrix=feature_matrix_raw,
                                                                  vectorizer=vectorizer,
                                                                  index=data.index,
                                                                  relevant_words=relevant_words)

    return feature_matrix


def predict(classifier, data, label=''):
    print(f'calculating predictions for for {label} SET...')
    return classifier.predict(data), classifier.predict_proba(data)[:, 1]


def analyze_performance(ground_truth, predictions, probabilities, label=''):
    print(f'------------------Performance Analysis for {label} SET Start--------------------')
    print(f'Accuracy = {accuracy_score(ground_truth, predictions)}')
    print(f'F1 Score = {f1_score(ground_truth, predictions)}')
    fpr, tpr, thresholds = roc_curve(ground_truth, probabilities, pos_label=1)
    area_under_roc_curve = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, area_under_roc_curve, label)
    print(f'Area under ROC curve = {area_under_roc_curve}')
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    print(f'TP = {tp}, FP = {fp}')
    print(f'FN = {fn}, TN = {tn}')
    print(f'------------------Performance Analysis for {label} SET End----------------------')


def plot_roc_curve(fpr, tpr, area, label):
    lw = 2
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {area:2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC for {label} SET')
    plt.legend(loc="lower right")
    plt.show()


def calculate_test_probability(test_case_index, X_matrix, words, classifier):
    feature_prob = pd.DataFrame(data=classifier.feature_log_prob_, columns=words).apply(np.exp)
    feature_count = pd.DataFrame(data=classifier.feature_count_, columns=words)
    prior_counts = classifier.class_count_
    prior_log_prob = classifier.class_log_prior_

    num_of_rows = len(X_matrix)
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


def get_n_most_used_words(n):
    return list(
        pd.read_csv(
            WORKING_DIRECTORY + '/resources/word_vectorization_matrices/rijen_prosinec/latest/occurrences_count_absolute.csv')[
            'Unnamed: 0'][:n])


def rename_columns_for_feature_matrix_with_index(matrix, vectorizer, index, relevant_words):
    return pd.DataFrame(data=matrix, index=index, columns=vectorizer.get_feature_names())[relevant_words]


def get_pctg_of_relevant_class(s):
    all_samples_count = len(s)
    relevant_samples_count = len(s[s == 1])
    irrelevant_samples_count = len(s[s == 0])
    pctg_rel = 100 * relevant_samples_count / all_samples_count
    return pctg_rel

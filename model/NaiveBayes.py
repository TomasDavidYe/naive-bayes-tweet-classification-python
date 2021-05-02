import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

from model.constants import RELEVANT, TEXT, TOP_50_WORD_LIST


class NaiveBayes:
    def __init__(self, posts: pd.DataFrame, stop_words: list):
        self.stop_words = stop_words
        self.posts = posts
        self.balanced_posts = self.get_balanced_post_dataset()

    def get_balanced_post_dataset(self):
        num_rows_relevant = len(self.posts[self.posts[RELEVANT] == 1])
        num_rows_non_relevant = len(self.posts[self.posts[RELEVANT] == 0])
        num_of_rows = min(num_rows_relevant, num_rows_non_relevant)

        sample_relevant = self.posts[self.posts[RELEVANT] == 1].iloc[:num_of_rows]
        sample_not_relevant = self.posts[self.posts[RELEVANT] == 0].iloc[:num_of_rows]

        balanced_dataset = pd.concat([sample_relevant, sample_not_relevant]).reset_index()
        return balanced_dataset

    def run_analysis(self, num_of_folds=3, classification_threshold=0.5, max_word_features=100):
        X = self.balanced_posts[TEXT]
        y = self.balanced_posts[RELEVANT]
        print(f'Percentage of relevant in FULL SET = {self.get_pctg_of_relevant_class(self.balanced_posts)}')
        skf = StratifiedKFold(n_splits=num_of_folds)
        fold_number = 0
        for train_index, test_index in skf.split(X, y):
            fold_number += 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.run_single_fold(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                fold_number=fold_number,
                threshold=classification_threshold,
                max_features=max_word_features
            )

    def run_single_fold(self, X_train, X_test, y_train, y_test, fold_number, threshold=0.5, max_features=100):
        print(f'-----------------------FOLD {fold_number} START-----------------------------------')
        print(f'Train set size = {len(X_train)}')
        print(f'Test set size = {len(X_test)}')
        label_train = f'TRAIN_FOLD_{fold_number}'
        label_test = f'TEST_FOLD_{fold_number}'
        print(f'Percentage of relevant in {label_train} SET = {self.get_pctg_of_relevant_class(y_train)}')
        print(f'Percentage of relevant in {label_test} SET = {self.get_pctg_of_relevant_class(y_test)}')

        vectorizer, features = self.fit_vectorizer(corpus=X_train, max_features=max_features)

        feature_matrix_train = self.build_feature_matrix(
            vectorizer=vectorizer,
            data=X_train,
            label=label_train,
            features=features
        )

        feature_matrix_test = self.build_feature_matrix(
            vectorizer=vectorizer,
            data=X_test,
            label=label_test,
            features=features
        )

        classifier = self.train(feature_matrix_train, y_train)

        train_predictions_default, train_predictions_threshold, train_probabilities = self.predict(
            classifier=classifier,
            data=feature_matrix_train,
            label=label_train,
            threshold=threshold
        )

        test_predictions_default, test_predictions_threshold, test_probabilities = self.predict(
            classifier=classifier,
            data=feature_matrix_test,
            label=label_test,
            threshold=threshold
        )

        # Default Threshold = 0.5
        self.analyze_performance(
            ground_truth=y_train,
            predictions=train_predictions_default,
            probabilities=train_probabilities,
            label=f'{label_train} with DEFAULT THRESHOLD = 0.5'
        )

        self.analyze_performance(
            ground_truth=y_test,
            predictions=test_predictions_default,
            probabilities=test_probabilities,
            label=f'{label_test} with DEFAULT THRESHOLD = 0.5'
        )

        # Selected Threshold
        self.analyze_performance(
            ground_truth=y_train,
            predictions=train_predictions_threshold,
            probabilities=train_probabilities,
            label=f'{label_train} with SELECTED THRESHOLD = {threshold}'
        )

        self.analyze_performance(
            ground_truth=y_test,
            predictions=test_predictions_threshold,
            probabilities=test_probabilities,
            label=f'{label_test} with SELECTED THRESHOLD = {threshold}'
        )

        print(f'-----------------------FOLD {fold_number} END-------------------------------------')

    def fit_vectorizer(self, corpus, max_features):
        print(f'Fitting vectorizer with MAX_WORD_FEATURES = {max_features}')
        vectorizer = CountVectorizer(stop_words=self.stop_words)
        vectorizer.fit(raw_documents=corpus)

        # TODO Remove the hardcoded list -> This is just to speed up the computation during refactoring
        # matrix = vectorizer.transform(corpus).todense()
        # full_feature_matrix = pd.DataFrame(data=matrix, index=corpus.index, columns=vectorizer.get_feature_names())
        # sorted_summed = full_feature_matrix.apply(func=np.sum, axis=0).sort_values(ascending=False)
        # column_list = list(sorted_summed[:max_features].index)

        column_list = TOP_50_WORD_LIST

        return vectorizer, column_list

    @staticmethod
    def train(features, target):
        print(f'Training model...')
        classifier = MultinomialNB()
        classifier.fit(X=features, y=target)

        return classifier

    def build_feature_matrix(self, vectorizer, data, features, label=''):
        print(f'Crating feature matrix for {label} SET...')
        feature_matrix_raw = vectorizer.transform(data).todense()
        feature_matrix = self.rename_columns_for_feature_matrix_with_index(
            matrix=feature_matrix_raw,
            vectorizer=vectorizer,
            index=data.index,
            features=features
        )

        return feature_matrix

    @staticmethod
    def predict(classifier, data, label='', threshold: float = 0.5):
        print(f'calculating predictions for for {label} SET...')
        probabilities: np.ndarray = classifier.predict_proba(data)[:, 1]
        default_predictions = classifier.predict(data)
        threshold_predictions = (probabilities > threshold).astype(int)
        return default_predictions, threshold_predictions, probabilities

    def analyze_performance(self, ground_truth, predictions, probabilities, label=''):
        print(f'------------------Performance Analysis for {label} SET Start--------------------')
        print(f'Accuracy = {accuracy_score(ground_truth, predictions)}')
        print(f'F1 Score = {f1_score(ground_truth, predictions)}')
        fpr, tpr, thresholds = roc_curve(ground_truth, probabilities, pos_label=1)
        # calculate the g-mean for each threshold
        gmeans = np.sqrt(tpr * (1 - fpr))
        # locate the index of the largest g-mean
        ix = np.argmax(gmeans)
        print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
        area_under_roc_curve = auc(fpr, tpr)
        self.plot_roc_curve(fpr, tpr, area_under_roc_curve, label)
        print(f'Area under ROC curve = {area_under_roc_curve}')
        tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
        print(f'TP = {tp}, FP = {fp}')
        print(f'FN = {fn}, TN = {tn}')
        print(f'------------------Performance Analysis for {label} SET End----------------------')

    @staticmethod
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

    @staticmethod
    def rename_columns_for_feature_matrix_with_index(matrix, vectorizer, index, features):
        full_feature_matrix = pd.DataFrame(
            data=matrix,
            index=index,
            columns=vectorizer.get_feature_names()
        )

        result = full_feature_matrix[features]
        return result

    @staticmethod
    def get_pctg_of_relevant_class(s):
        all_samples_count = len(s)
        relevant_samples_count = len(s[s == 1])
        irrelevant_samples_count = len(s[s == 0])
        pctg_rel = 100 * relevant_samples_count / all_samples_count
        return pctg_rel

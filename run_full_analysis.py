import pandas as pd

from model.NaiveBayes import NaiveBayes


def get_stop_words():
    stop_words = pd.read_csv('data/stop_words.csv', header=None)
    result = list(set(stop_words[0]))
    result.sort()
    return result


def get_posts():
    return pd.read_csv('./data/posts.csv')


if __name__ == '__main__':
    NaiveBayes(
        posts=get_posts(),
        stop_words=get_stop_words()
    ).run_analysis(
        num_of_folds=2,
        classification_threshold=0.65,
        max_word_features=50
    )

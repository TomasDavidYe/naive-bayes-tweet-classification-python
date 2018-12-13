import pandas as pd


def get_stop_words_new():
    stop_words_new = pd.read_excel('../resources/general_data/stop_words_13_12_2018.xlsx')['word'].to_dict().values()
    result = list(set(stop_words_new))
    result.sort()
    return result


def save_stopwords_to_file():
    pd.Series(data=get_stop_words_new(), name='Stop_words').to_csv('resources/stop_words.csv')
from stop_words import get_stop_words
from load_data import clear_diacritics_from
import pandas as pd

def get_custom_stop_words():
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
    stop_words_cz.sort()
    return stop_words_cz


def save_stopwords_to_file():
    pd.Series(data=get_custom_stop_words(), name='Stop_words').to_csv('resources/stop_words.csv')
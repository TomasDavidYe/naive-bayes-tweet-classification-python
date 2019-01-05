import pandas as pd


def get_columns_for_word_aggregate_matrix():
    return [
        'word_text',
        'word_count_zminka',
        'word_count_zminka_rel',
        'word_count_zminka_nerel',
        'word_zminka_rank',
        'word_zminka_rank_rel',
        'word_zminka_rank_nerel'
    ]


def get_stop_words():
    stop_words_new = pd.read_excel('../resources/general_data/stop_words_13_12_2018.xlsx')['word'].to_dict().values()
    result = list(set(stop_words_new))
    result.sort()
    return result





def get_column_names_for_simple_matrix_dictionary():
    return {
        'domain': 'Doména',
        'domaingroup': 'Skupina domén',
    }

def get_replace_dictionary():
    return {
        'č': 'c',
        'š': 's',
        'ř': 'r',
        'ž': 'z',
        'ě': 'e',
        'é': 'e',
        'ý': 'y',
        'á': 'a',
        'í': 'i',
        'ó': 'o',
        'ů': 'u',
        'ú': 'u',
        'ň': 'n',
        'ť': 't',
        'ď': 'd',
    }


def get_column_names_for_dropping():
    return [
        'Unnamed: 0',
        'Obsah zmínek',
        'Vloženo do databáze',
        'Kontext',
        'Odkaz na zmínku',
        'Doména',
        'Sentiment',
        'Index sentimentu',
        'Skupina domén',
        'Klíčová slova',
        'Pohlaví',
        'Geolokace'
    ]


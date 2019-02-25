import pandas as pd
import re
from helper_methods import get_replace_dictionary
import datetime




def clear_diacritics_from(sentence):
    replace_dictionary = get_replace_dictionary()
    if isinstance(sentence, str):
        sentence = sentence.lower()
        for key in replace_dictionary.keys():
            sentence = sentence.replace(key, replace_dictionary[key])
        return sentence
    else:
        return ''


def clear_diacritics_from_columns(data, column_names):
    regex = re.compile('[^a-zA-Z ]')
    result = data.copy()
    for column_name in column_names:
        result[column_name] = result[column_name].apply(lambda x: regex.sub('', clear_diacritics_from(x)))
    return result


def map_category_to_relevance(category, categories):
    relevant_categories = categories - set(['nerelevantní'])
    if category in relevant_categories:
        return 'rel'
    else:
        return 'nerel'


def load_and_clean_data(file_name):
    path = '../resources/source_data/data_' + file_name
    dataset = pd.read_excel(path)
    columns_names_with_text = ['Obsah zmínek', 'Kontext', 'Klíčová slova']
    dataset = clear_diacritics_from_columns(data=dataset, column_names=columns_names_with_text)
    categories = set(dataset['Štítek'].unique())
    column_name_for_dropping = [ 'Druh', 'Titul', 'Body kvality', 'Název projektu', 'Kategorie domény']
    dataset.drop(columns=column_name_for_dropping, inplace=True)
    dataset['Štítek'] = dataset['Štítek'].apply(lambda x: map_category_to_relevance(x, categories))
    return dataset


def save_cleaner_data(filename):
    month = filename.split('.')[0]
    path = '../resources/source_data/cleaner_data_' + month + '.csv'
    load_and_clean_data(filename).to_csv(path)

save_cleaner_data('prosinec.xlsx')
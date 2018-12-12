import pandas as pd
import re

regex = re.compile('[^a-zA-Z ]')
replace_dictionary = {
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


def clear_diacritics_from(sentence):
    if isinstance(sentence, str):
        sentence = sentence.lower()
        for key in replace_dictionary.keys():
            sentence = sentence.replace(key, replace_dictionary[key])
        return regex.sub('', sentence)




def clear_diacritics_from_columns(data, column_names):
    result = data.copy()
    for column_name in column_names:
        result[column_name] = result[column_name].apply(lambda x: clear_diacritics_from(x))
    return result


def map_category_to_relevance(category):
    categories = set(dataset['Štítek'].unique())
    relevant_categories = categories - set(['nerelevantní'])
    if category in relevant_categories:
        return 'rel'
    else:
        return 'nerel'


dataset = pd.read_excel('resources/general_data/data.xlsm')
columns_names_with_text = ['Obsah zmínek', 'Kontext']
dataset = clear_diacritics_from_columns(data=dataset, column_names=columns_names_with_text)
column_name_for_dropping = ['id', 'Druh', 'Titul', 'Body kvality', 'Název projektu', 'Kategorie domény']
dataset.drop(columns=column_name_for_dropping, inplace=True)
dataset['Štítek'] = dataset['Štítek'].apply(lambda x: map_category_to_relevance(x))

dataset.to_csv('resources/general_data/cleaner_data.csv')

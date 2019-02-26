import pandas as pd
import re
from helper_methods import get_replace_dictionary


def clear_duplicities(data: pd.DataFrame):
    result: pd.DataFrame = data.copy()
    result.dropna(subset=['Obsah zmínek'])

    twitter_retweet_regex = re.compile('rt [a-z]* ')
    result['Obsah zmínek'] = result['Obsah zmínek'].apply(lambda x: twitter_retweet_regex.sub('', str(x)))

    redundant_space_regex = re.compile('^ +')
    result['Obsah zmínek'] = result['Obsah zmínek'].apply(lambda x: redundant_space_regex.sub('', str(x)))

    result.sort_values(by='Datum vytvoření', inplace=True)
    result.drop_duplicates(subset=['Obsah zmínek'], keep='first', inplace=True)

    return result


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
    return dataset.dropna(subset=['Obsah zmínek'])


def save_cleaner_data(filename):
    month = filename.split('.')[0]
    path = '../resources/source_data/cleaner_data_' + month + '.csv'
    load_and_clean_data(filename).to_csv(path)



# ratio = 0.35
# data = load_and_clean_data('rijen_prosinec.xlsx')
# train_length = int(len(data) * ratio)
#
# train_set = data.iloc[:train_length]
# test_set = data.iloc[train_length:]
# train_set.iloc[-1]['id']

# save_cleaner_data('prosinec.xlsx')
# save_cleaner_data('rijen.xlsm')

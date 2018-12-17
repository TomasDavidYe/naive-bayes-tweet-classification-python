import pandas as pd

def get_stop_words_new():
    stop_words_new = pd.read_excel('resources/general_data/stop_words_13_12_2018.xlsx')['word'].to_dict().values()
    result = list(set(stop_words_new))
    result.sort()
    return result


def create_simple_aggregate_matrix_for(variable_name):
    column_name = get_column_names_for_simple_matrix_dictionary().get(variable_name)
    data = pd.read_csv('resources/general_data/cleaner_data.csv')[[column_name, 'Štítek']].dropna(subset=[column_name])
    value_counts = data[column_name].value_counts()
    values = value_counts.keys()
    temp = list([])
    for value in values:
        row = dict({})
        value_data = data.loc[data[column_name] == value]['Štítek']
        row[variable_name] = value
        row[variable_name + '_count'] = len(value_data)
        row[variable_name + '_count_rel'] = len(value_data.loc[value_data == 'rel'])
        row[variable_name + '_count_nerel'] = len(value_data.loc[value_data == 'nerel'])
        temp.append(row)
    aggregate_matrix = pd.DataFrame(data=temp)[[variable_name, variable_name + '_count', variable_name + '_count_rel', variable_name + '_count_nerel']]
    return aggregate_matrix


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
        'Datum vytvoření',
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


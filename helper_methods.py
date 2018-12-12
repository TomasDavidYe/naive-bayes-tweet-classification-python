import pandas as pd


def create_simple_aggregate_matrix_for(variable_name):
    column_name = get_column_name_dictionary().get(variable_name)
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


def get_column_name_dictionary():
    return {
        'domain': 'Doména',
        'domaingroup': 'Skupina domén',
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



A = pd.DataFrame(data=[[1, 2],
                       [4, 5],
                       [7, 8]],
                 index=['R1', 'R2', 'R3'],
                 columns=['C1', 'C2'])


A.shape
len(A)
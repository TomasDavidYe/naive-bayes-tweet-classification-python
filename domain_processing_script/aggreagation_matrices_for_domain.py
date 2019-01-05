from helper_methods import get_column_names_for_simple_matrix_dictionary
from datetime import datetime
import pandas as pd


def create_simple_aggregate_matrix_for(variable_name, ratio):
    column_name = get_column_names_for_simple_matrix_dictionary().get(variable_name)
    data = pd.read_csv('../resources/general_data/cleaner_data.csv')[[column_name, 'Štítek', 'Datum vytvoření']].dropna(subset=[column_name])
    data.sort_values(by='Datum vytvoření', inplace=True)
    data.drop(columns=['Datum vytvoření'])
    train_set_size = int(len(data) * ratio)
    data = data[:train_set_size]
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


def create_aggregation_matrices_for_domain_and_domain_group(ratio=1.0):
    time = datetime.now().strftime('%c').replace(' ', '_')
    aggregate_matrix_domain = create_simple_aggregate_matrix_for('domain', ratio)
    aggregate_matrix_domaingroup = create_simple_aggregate_matrix_for('domaingroup', ratio)
    aggregate_matrix_domain.to_csv(
        '../resources/aggregation_matrices/domain/aggregation_matrix_domain_' + time + '.csv')
    aggregate_matrix_domain.to_csv('../resources/aggregation_matrices/domain/latest.csv')
    aggregate_matrix_domaingroup.to_csv(
        '../resources/aggregation_matrices/domain_group/aggregation_matrix_domaingroup_' + time + '.csv')
    aggregate_matrix_domaingroup.to_csv(
        '../resources/aggregation_matrices/domain_group/latest.csv')


create_aggregation_matrices_for_domain_and_domain_group(0.75)
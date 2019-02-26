import time
import os
from preprocessing_scripts.load_data import save_cleaner_data
from authors_processing_script.aggregation_matrix_for_author import create_aggregation_matrix_author
from domain_processing_script.aggreagation_matrices_for_domain import create_aggregation_matrices_for_domain_and_domain_group
from word_processing_script.word_pre_processing import vectorize
from feature_processing_scripts.create_full_feature_matrix import build_all_feature_matrices_for_given_month


def create_directory_for_files(month):
    pass


def create_feature_matrices_from_file(filename, ratio):
    month = filename.split('.')[0]
    create_directory_for_files(month)
    os.chdir(os.getcwd() + '/preprocessing_scripts')
    save_cleaner_data(filename)

    print('Waiting to save all the matrices correctly...')
    time.sleep(3)

    os.chdir(os.path.join(os.getcwd(), '..'))
    os.chdir(os.getcwd() + '/authors_processing_script')
    create_aggregation_matrix_author(month, ratio)

    os.chdir(os.path.join(os.getcwd(), '..'))
    os.chdir(os.getcwd() + '/domain_processing_script')
    create_aggregation_matrices_for_domain_and_domain_group(month, ratio)

    os.chdir(os.path.join(os.getcwd(), '..'))
    os.chdir(os.getcwd() + '/word_processing_script')
    vectorize(month, ratio)

    print('Waiting to save all the matrices correctly...')
    time.sleep(3)

    os.chdir(os.path.join(os.getcwd(), '..'))
    os.chdir(os.getcwd() + '/feature_processing_scripts')
    build_all_feature_matrices_for_given_month(filename)
    os.chdir(os.path.join(os.getcwd(), '..'))


create_feature_matrices_from_file('rijen.xlsm', 0.7)
create_feature_matrices_from_file('prosinec.xlsx', 0.7)


import pandas as pd
import os
os.getcwd()
rijen = pd.read_excel('./resources/source_data/data_rijen.xlsm')
prosinec = pd.read_excel('./resources/source_data/data_prosinec.xlsx')
rijen_prosinec = rijen.append(prosinec, ignore_index=True).set_index('id')
rijen_prosinec.to_excel('./resources/source_data/data_prosinec_rijen.xlsx')
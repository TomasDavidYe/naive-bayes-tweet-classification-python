import time
import os

from constants import WORKING_DIRECTORY
from preprocessing_scripts.load_data import save_cleaner_data
from authors_processing_script.aggregation_matrix_for_author import create_aggregation_matrix_author
from domain_processing_script.aggreagation_matrices_for_domain import \
    create_aggregation_matrices_for_domain_and_domain_group
from word_processing_script.word_pre_processing import vectorize
from feature_processing_scripts.create_full_feature_matrix import build_all_feature_matrices_for_given_month


def create_directories_for_data_files(month):
    folder_names = ['/aggregation_matrices/authors/',
                    '/aggregation_matrices/domain/',
                    '/aggregation_matrices/domain_group/',
                    '/feature_matrices/',
                    '/word_vectorization_matrices/']

    if not os.path.exists(WORKING_DIRECTORY + '/aggregation_matrices/'):
        os.mkdir(WORKING_DIRECTORY + '/aggregation_matrices/')

    for folder_name in folder_names:
        if not os.path.exists(WORKING_DIRECTORY + '/resources' + folder_name):
            os.mkdir(WORKING_DIRECTORY + '/resources' + folder_name)
        if not os.path.exists(WORKING_DIRECTORY + '/resources' + folder_name + month):
            os.mkdir(WORKING_DIRECTORY + '/resources' + folder_name + month)
    if not os.path.exists(WORKING_DIRECTORY + '/resources/word_vectorization_matrices/' + month + '/latest'):
        os.mkdir(WORKING_DIRECTORY + '/resources/word_vectorization_matrices/' + month + '/latest')


def create_feature_matrices_from_file(filename, ratio):
    month = filename.split('.')[0]
    create_directories_for_data_files(month)

    save_cleaner_data(filename)

    print('Waiting to save all the matrices correctly...')
    time.sleep(3)

    create_aggregation_matrix_author(month, ratio)


    create_aggregation_matrices_for_domain_and_domain_group(month, ratio)

    vectorize(month, ratio)

    print('Waiting to save all the matrices correctly...')
    time.sleep(3)

    build_all_feature_matrices_for_given_month(filename)


create_feature_matrices_from_file('rijen_prosinec.xlsx', 0.80)

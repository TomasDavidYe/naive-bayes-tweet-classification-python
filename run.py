from general_data_scripts.load_data import load_and_clean_data


def create_feature_matrices_from_file(filename):
    clean_data = load_and_clean_data(filename)tt
import pandas as pd

from constants import WORKING_DIRECTORY, TEXT, RELEVANT
from ml_model.ml_utils import run_optimisation


def run(num_folds: int, decision_threshold: float):
    full_dataset = pd.read_csv(WORKING_DIRECTORY + '/resources/source_data/cleaner_data_rijen_prosinec.csv')
    data = full_dataset[['id']].copy()
    data[TEXT] = full_dataset['Obsah zmínek']
    data[RELEVANT] = full_dataset['Štítek'].map(lambda x: int(x == 'rel'))
    data = data.drop_duplicates(subset=['id'])
    data.dropna(inplace=True)

    num_of_rows = 4000
    sample_relevant = data[data[RELEVANT] == 1].iloc[:num_of_rows]
    sample_not_relevant = data[data[RELEVANT] == 0].iloc[:num_of_rows]

    optimisation_data = pd.concat([sample_relevant, sample_not_relevant]).reset_index()

    run_optimisation(
        data=optimisation_data,
        num_of_folds=num_folds,
        threshold=decision_threshold
    )

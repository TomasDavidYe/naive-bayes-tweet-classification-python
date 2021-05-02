import pandas as pd

from constants import RELEVANT
from ml_model.ml_utils import run_optimisation


def run(num_folds: int, classification_threshold: float, max_word_features: int):
    num_of_rows = 4000
    data = pd.read_csv('./resources/source_data/posts.csv')

    sample_relevant = data[data[RELEVANT] == 1].iloc[:num_of_rows]
    sample_not_relevant = data[data[RELEVANT] == 0].iloc[:num_of_rows]

    optimisation_data = pd.concat([sample_relevant, sample_not_relevant]).reset_index()

    run_optimisation(
        data=optimisation_data,
        num_of_folds=num_folds,
        threshold=classification_threshold,
        max_features=max_word_features,
    )

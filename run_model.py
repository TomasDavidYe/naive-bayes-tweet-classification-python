from ml_model.naive_bayes import run

if __name__ == '__main__':
    run(
        num_folds=3,
        classification_threshold=0.65
    )

from ml_model.naive_bayes import run

if __name__ == '__main__':
    run(
        num_folds=3,
        decision_threshold=0.5
    )

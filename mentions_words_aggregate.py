import pandas as pd


ABSOLUTE_FREQUENCY = 'absolute_frequency'
WORD_TEXT = 'word_text'
MENTIONS_CSV = 'counts_in_mentions.csv'
STITEK = 'Štítek'
OBSAH_ZMINKY = 'Obsah zmínek'
CLEANER_DATA_CSV = 'cleaner_data.csv'

dataset = pd.read_csv(CLEANER_DATA_CSV)
mentions_with_relevance = dataset[[OBSAH_ZMINKY, STITEK]].dropna()
words_in_mentions = pd.read_csv(MENTIONS_CSV)
words_in_mentions.columns = [WORD_TEXT, ABSOLUTE_FREQUENCY]



def check_if_word_is_in_text(text, word):
    if not isinstance(text, str):
        return False
    seq = text.split()
    return word in seq


def check_if_word_is_in_relevant_text(text, word, relevance):
    return check_if_word_is_in_text(text, word) and relevance == 'rel'


def check_if_word_is_in_irrelevant_text(text, word, relevance):
    return check_if_word_is_in_text(text, word) and relevance == 'nerel'


def get_word_count_in_rel_mentions(word):
    selection = mentions_with_relevance.apply(func=lambda x: check_if_word_is_in_relevant_text(word=word,
                                                                                               text=x[OBSAH_ZMINKY],
                                                                                               relevance=x[STITEK]),
                                              axis=1)
    return len(mentions_with_relevance.loc[selection, :])


def get_word_count_in_nerel_mentions(word):
    selection = mentions_with_relevance.apply(func=lambda x: check_if_word_is_in_irrelevant_text(word=word,
                                                                                                 text=x[OBSAH_ZMINKY],
                                                                                                 relevance=x[STITEK]),
                                              axis=1)
    return len(mentions_with_relevance.loc[selection, :])

def get_aggreate_matrix_with_first_n_words(n):
    AGGREGATE_COLUMNS = ['rank', 'word_text', 'absolute_count', 'word_count_zminka_rel', 'word_count_zminka_nerel']
    top_words = words_in_mentions.loc[:n, :]
    aggregate_matrix = pd.DataFrame(data=[], columns=AGGREGATE_COLUMNS)
    aggregate_matrix.shape
    word = None
    for rank, word_field in enumerate(top_words.values):
        word = list(word_field)[0]
        count = list(word_field)[1]
        row = pd.Series(index=AGGREGATE_COLUMNS,
                        data=[rank + 1, word, count, get_word_count_in_rel_mentions(word),
                              get_word_count_in_nerel_mentions(word)],
                        )
        aggregate_matrix = aggregate_matrix.append(row, ignore_index=True)



get_aggreate_matrix_with_first_n_words(100)
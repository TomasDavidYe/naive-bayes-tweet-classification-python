import pandas as pd
from datetime import datetime
import re
from sklearn.feature_extraction.text import CountVectorizer

# import os
# os.chdir(os.getcwd() + '/feature_processing_scripts')

class FeatureMatrixBuilder:

    def __init__(self, source):
        self.matrix = pd.read_csv(source).dropna(subset=['Obsah zmínek'])[['id', 'Datum vytvoření']].set_index('id')

    def build(self):
        return self.matrix

    def save(self, name):
        print('Saving...')
        regex = re.compile('[/ ]')
        time = regex.sub('_', datetime.now().strftime('%c').lower())
        self.matrix.to_csv('../resources/feature_matrices/'+name+'_feature_matrix_' + time + '.csv')


    # TODO refactor it so it doesn contain duplicates
    def add_author_features(self):
        print('Adding author features...')
        authors_db = pd.read_csv('../resources/aggregation_matrices/authors/latest.csv').drop(columns=['Unnamed: 0']).set_index('author')
        empty_author = pd.Series(name='XXXNONAMEXXX', data=[0 for column_name in authors_db.columns.values], index=authors_db.columns.values)
        authors_db = authors_db.append(empty_author)
        id_to_author = pd.read_csv('../resources/general_data/cleaner_data.csv').dropna(subset=['Obsah zmínek'])[['id', 'Autor']].set_index('id').fillna('XXXNONAMEXXX')
        temp = {}
        for ident in id_to_author.index:
            author = id_to_author.loc[ident, 'Autor']
            author_row = authors_db.loc[author]
            temp[ident] = dict(author_row)
        data = pd.DataFrame(data=temp).transpose()
        data.loc[:, 'author_percentage_rel'] = (data.loc[:, 'author_count_rel'] + 1) / (data.loc[:, 'author_count_nerel'] + 1)
        self.matrix = self.matrix.join(data)
        return self

    def add_domain_features(self):
        print('Adding domain features...')
        domain_db = pd.read_csv('../resources/aggregation_matrices/domain/latest.csv').drop(columns=['Unnamed: 0']).set_index('domain')
        id_to_domain = pd.read_csv('../resources/general_data/cleaner_data.csv')[['id', 'Doména']].set_index('id')
        temp = {}
        for ident in id_to_domain.index:
            domain = id_to_domain.loc[ident, 'Doména']
            domain_row = domain_db.loc[domain]
            temp[ident] = dict(domain_row)
        data = pd.DataFrame(data=temp).transpose()
        data.loc[:, 'domain_percentage_rel'] = (data.loc[:, 'domain_count_rel'] + 1) / (data.loc[:, 'domain_count_nerel'] + 1)
        self.matrix = self.matrix.join(data)
        return self

    def add_domain_group_features(self):
        print('Adding domain_group features...')
        domain_group_db = pd.read_csv('../resources/aggregation_matrices/domain_group/latest.csv').drop(columns=['Unnamed: 0']).set_index('domaingroup')
        id_to_domain_group = pd.read_csv('../resources/general_data/cleaner_data.csv')[['id', 'Skupina domén']].set_index('id')
        temp = {}
        for ident in id_to_domain_group.index:
            domain_group = id_to_domain_group.loc[ident, 'Skupina domén']
            domain_group_row = domain_group_db.loc[domain_group]
            temp[ident] = dict(domain_group_row)
        data = pd.DataFrame(data=temp).transpose()
        data.loc[:, 'domaingroup_percentage_rel'] = (data.loc[:, 'domaingroup_count_rel'] + 1) / (data.loc[:, 'domaingroup_count_nerel'] + 1)
        self.matrix = self.matrix.join(data)
        return self

    def add_word_count_features(self):
        print('Adding word count features...')
        feature_matrix = self.get_feature_matrix()
        self.matrix = self.matrix.join(feature_matrix)
        return self

    def add_word_identicator_features(self):
        print('Adding word indicator features...')
        feature_matrix = self.get_feature_matrix()
        id_matrix = feature_matrix.applymap(lambda x: int(x > 0))
        self.matrix = self.matrix.join(id_matrix)
        return self

    def get_frequency_count_all_features(self):
        return self

    def get_frequency_count_rel_features(self):
        return self

    def get_frequency_count_nerel_features(self):
        return self

    def add_mention_word_count_feature(self):
        data = pd.read_csv('../resources/general_data/cleaner_data.csv')[['id', 'Obsah zmínek']].dropna(
            subset=['Obsah zmínek']).set_index('id')
        data.columns = ['other_zminka_count_words']
        transformed = data['other_zminka_count_words'].map(lambda x: len(x.split(' ')))
        self.matrix = self.matrix.join(transformed)
        return self

    def add_link_count_feature(self):
        regex = re.compile('https:\\/\\/')
        data = pd.read_excel('../resources/general_data/data.xlsm')[['id', 'Obsah zmínek']].dropna(
            subset=['Obsah zmínek']).set_index('id')
        data.columns = ['other_zminka_count_links']
        transformed = data['other_zminka_count_links'].map(lambda x: len(regex.findall(x)))
        self.matrix = self.matrix.join(transformed)
        return self

    def add_mention_letter_count_feature(self):
        data = pd.read_csv('../resources/general_data/cleaner_data.csv')[['id', 'Obsah zmínek']].dropna(
            subset=['Obsah zmínek']).set_index('id')
        data.columns = ['other_zminka_count_letters']
        transformed = data['other_zminka_count_letters'].map(lambda x: len(x))
        self.matrix = self.matrix.join(transformed)
        return self

    def add_mention_hour_feature(self):
        data = pd.read_csv('../resources/general_data/cleaner_data.csv').dropna(
            subset=['Obsah zmínek'])[['id', 'Datum vytvoření']].set_index('id')
        data.columns = ['other_time_hour']
        transformed = data['other_time_hour'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M').hour)
        self.matrix = self.matrix.join(transformed)
        return self

    def add_mention_weekday_feature(self):
        data = pd.read_csv('../resources/general_data/cleaner_data.csv').dropna(
            subset=['Obsah zmínek'])[['id', 'Datum vytvoření']].set_index('id')
        data.columns = ['other_time_weekday']
        transformed = data['other_time_weekday'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M').weekday())
        self.matrix = self.matrix.join(transformed)
        return self

    def add_indicator_of_diacritics_usage(self):
        regex = re.compile('[ěščřžýáíéúůňťďĚŠČŘŽÝÁÍÉÚŮŇĎŤ]')
        data = pd.read_excel('../resources/general_data/data.xlsm')[['id', 'Obsah zmínek']].dropna(
            subset=['Obsah zmínek']).set_index('id')
        data.columns = ['other_zminka_diacritic_usage']
        transformed = data['other_zminka_diacritic_usage'].map(lambda x: int(regex.search(x) is not None))
        self.matrix = self.matrix.join(transformed)
        return self

    def get_feature_matrix(self):
        id_to_mention = pd.read_csv('../resources/general_data/cleaner_data.csv').dropna(subset=['Obsah zmínek'])[
            ['id', 'Obsah zmínek']]
        vocabulary = list(pd.read_csv('../resources/word_vectorization_matrices/latest/occurrences_count_all.csv',
                                      names=['word_text', 'count']).loc[:5000, 'word_text'])
        vectorizer = CountVectorizer(vocabulary=vocabulary)
        return pd.DataFrame(index=id_to_mention['id'], columns=vocabulary,
                            data=vectorizer.transform(id_to_mention['Obsah zmínek']).todense())



builder = FeatureMatrixBuilder(source='../resources/general_data/cleaner_data.csv')
result = builder.add_mention_word_count_feature()\
    .add_mention_letter_count_feature()\
    .add_link_count_feature()\
    .add_mention_weekday_feature()\
    .add_mention_hour_feature()\
    .add_indicator_of_diacritics_usage()\
    .build()

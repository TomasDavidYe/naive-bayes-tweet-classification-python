import pandas as pd
from datetime import datetime
import re
from sklearn.feature_extraction.text import CountVectorizer

# import os
# os.chdir(os.getcwd() + '/feature_processing_scripts')
from constants import WORKING_DIRECTORY


class FeatureMatrixBuilder:

    def __init__(self, filename):
        self.filename = filename
        self.month = filename.split('.')[0]
        self.reset_matrix()

    def build(self):
        return self.matrix

    def save(self, name):
        print('Saving...')
        time = datetime.now().strftime('%d_%m_%Y')
        save_path = WORKING_DIRECTORY + '/resources/feature_matrices/' + self.month + '/' + name + '_feature_matrix_' + time + '.csv'
        self.matrix.sort_values(by='Datum vytvoření').to_csv(save_path)

    def reset_matrix(self):
        source = WORKING_DIRECTORY + '/resources/source_data/cleaner_data_' + self.month + '.csv'
        self.matrix = pd.read_csv(source).dropna(subset=['Obsah zmínek'])[['id', 'Datum vytvoření']].set_index('id')

    # TODO refactor it so it doesn contain duplicates
    def add_author_features(self):
        print('Adding author features...')
        source_path = WORKING_DIRECTORY + '/resources/aggregation_matrices/authors/' + self.month + '/latest.csv'
        authors_db = pd.read_csv(source_path).drop(columns=['Unnamed: 0']).set_index('author')
        empty_author = pd.Series(name='XXXNONAMEXXX', data=[-10 for column_name in authors_db.columns.values],
                                 index=authors_db.columns.values)
        authors_db = authors_db.append(empty_author)
        id_to_author = \
        pd.read_csv(WORKING_DIRECTORY + '/resources/source_data/cleaner_data_' + self.month + '.csv').dropna(
            subset=['Obsah zmínek'])[['id', 'Autor']].set_index('id').fillna('XXXNONAMEXXX')
        known_authors = authors_db.index
        temp = {}
        for ident in id_to_author.index:
            author = id_to_author.loc[ident, 'Autor']
            if type(author) is pd.Series:
                temp[ident] = dict(pd.Series(data=[-10 for column_name in authors_db.columns.values],
                                             index=authors_db.columns.values))
            else:
                if author in known_authors:
                    author_row = authors_db.loc[author]
                    temp[ident] = dict(author_row)
                else:
                    temp[ident] = dict(pd.Series(data=[-10 for column_name in authors_db.columns.values],
                                                 index=authors_db.columns.values))
        data = pd.DataFrame(data=temp).transpose()
        data.loc[:, 'author_percentage_rel'] = (data.loc[:, 'author_count_rel'] + 1) / (
                    data.loc[:, 'author_count_nerel'] + 1)
        self.matrix = self.matrix.join(data)
        return self

    def add_domain_features(self):
        print('Adding domain features...')
        source_path = WORKING_DIRECTORY + '/resources/aggregation_matrices/domain/' + self.month + '/latest.csv'
        domain_db = pd.read_csv(source_path).drop(columns=['Unnamed: 0']).set_index('domain')
        id_to_domain = pd.read_csv(WORKING_DIRECTORY + '/resources/source_data/cleaner_data_' + self.month + '.csv')[
            ['id', 'Doména']].set_index('id')
        known_domains = list(domain_db.index)
        temp = {}

        for ident in id_to_domain.index:
            domain = id_to_domain.loc[ident, 'Doména']
            if type(domain) is pd.Series:
                temp[ident] = {'domain_count': -10, 'domain_count_rel': -10, 'domain_count_nerel': -10}
            else:
                if domain in known_domains:
                    domain_row = domain_db.loc[domain]
                    temp[ident] = dict(domain_row)
                else:
                    temp[ident] = {'domain_count': -10, 'domain_count_rel': -10, 'domain_count_nerel': -10}

        data = pd.DataFrame(data=temp).transpose()
        data.loc[:, 'domain_percentage_rel'] = (data.loc[:, 'domain_count_rel'] + 1) / (
                    data.loc[:, 'domain_count_nerel'] + 1)
        self.matrix = self.matrix.join(data)
        return self

    def add_domain_group_features(self):
        print('Adding domain_group features...')
        domain_group_db = pd.read_csv(
            WORKING_DIRECTORY + '/resources/aggregation_matrices/domain_group/' + self.month + '/latest.csv').drop(
            columns=['Unnamed: 0']).set_index('domaingroup')
        known_domain_groups = list(domain_group_db.index)
        id_to_domain_group = \
        pd.read_csv(WORKING_DIRECTORY + '/resources/source_data/cleaner_data_' + self.month + '.csv')[
            ['id', 'Skupina domén']].set_index('id')
        temp = {}

        for ident in id_to_domain_group.index:
            domain_group = id_to_domain_group.loc[ident, 'Skupina domén']
            if type(domain_group) is pd.Series:
                temp[ident] = {'domaingroup_count': -10, 'domaingroup_count_rel': -10,
                               'domaingroup_count_nerel': -10}
            else:
                if domain_group in known_domain_groups:
                    domain_group_row = domain_group_db.loc[domain_group]
                    temp[ident] = dict(domain_group_row)
                else:
                    temp[ident] = {'domaingroup_count': -10, 'domaingroup_count_rel': -10,
                                   'domaingroup_count_nerel': -10}

        data = pd.DataFrame(data=temp).transpose()
        data.loc[:, 'domaingroup_percentage_rel'] = (data.loc[:, 'domaingroup_count_rel'] + 1) / (
                    data.loc[:, 'domaingroup_count_nerel'] + 1)
        self.matrix = self.matrix.join(data)
        return self

        A = domain_group_db.loc['Facebook']

    def add_word_count_features(self):
        print('Adding word count features...')
        feature_matrix = self.get_feature_matrix()
        self.matrix = self.matrix.join(feature_matrix)
        return self

    def add_word_indicator_features(self):
        print('Adding word indicator features...')
        feature_matrix = self.get_feature_matrix()
        id_matrix = feature_matrix.applymap(lambda x: int(x > 0))
        self.matrix = self.matrix.join(id_matrix)
        return self

    # TODO implement these methods!!!
    def get_frequency_count_all_features(self):
        return self

    def get_frequency_count_rel_features(self):
        return self

    def get_frequency_count_nerel_features(self):
        return self

    def add_mention_word_count_feature(self):
        data = pd.read_csv(WORKING_DIRECTORY + '/resources/source_data/cleaner_data_' + self.month + '.csv')[
            ['id', 'Obsah zmínek']].dropna(
            subset=['Obsah zmínek']).set_index('id')
        data.columns = ['other_zminka_count_words']
        transformed = data['other_zminka_count_words'].map(lambda x: len(x.split(' ')))
        self.matrix = self.matrix.join(transformed)
        return self

    def add_link_count_feature(self):
        regex = re.compile('https:\\/\\/')

        data = pd.read_excel(WORKING_DIRECTORY + '/resources/source_data/data_' + self.filename)[
            ['id', 'Obsah zmínek']].dropna(
            subset=['Obsah zmínek']).set_index('id')
        data.columns = ['other_zminka_count_links']
        transformed = data['other_zminka_count_links'].map(lambda x: len(regex.findall(x)))
        self.matrix = self.matrix.join(transformed)
        return self

    def add_mention_letter_count_feature(self):
        data = pd.read_csv(WORKING_DIRECTORY + '/resources/source_data/cleaner_data_' + self.month + '.csv')[
            ['id', 'Obsah zmínek']].dropna(
            subset=['Obsah zmínek']).set_index('id')
        data.columns = ['other_zminka_count_letters']
        transformed = data['other_zminka_count_letters'].map(lambda x: len(x))
        self.matrix = self.matrix.join(transformed)
        return self

    def add_mention_hour_feature(self):
        data = pd.read_csv(WORKING_DIRECTORY + '/resources/source_data/cleaner_data_' + self.month + '.csv').dropna(
            subset=['Obsah zmínek'])[['id', 'Datum vytvoření']].set_index('id')
        data.columns = ['other_time_hour']
        transformed = data['other_time_hour'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M').hour)
        self.matrix = self.matrix.join(transformed)
        return self

    def add_mention_weekday_feature(self):
        data = pd.read_csv(WORKING_DIRECTORY + '/resources/source_data/cleaner_data_' + self.month + '.csv').dropna(
            subset=['Obsah zmínek'])[['id', 'Datum vytvoření']].set_index('id')
        data.columns = ['other_time_weekday']
        transformed = data['other_time_weekday'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M').weekday())
        self.matrix = self.matrix.join(transformed)
        return self

    def add_indicator_of_diacritics_usage(self):
        regex = re.compile('[ěščřžýáíéúůňťďĚŠČŘŽÝÁÍÉÚŮŇĎŤ]')

        data = pd.read_excel(WORKING_DIRECTORY + '/resources/source_data/data_' + self.filename)[
            ['id', 'Obsah zmínek']].dropna(
            subset=['Obsah zmínek']).set_index('id')
        data.columns = ['other_zminka_diacritic_usage']
        transformed = data['other_zminka_diacritic_usage'].map(lambda x: int(regex.search(x) is not None))
        self.matrix = self.matrix.join(transformed)
        return self

    def get_feature_matrix(self):
        id_to_mention = \
        pd.read_csv(WORKING_DIRECTORY + '/resources/source_data/cleaner_data_' + self.month + '.csv').dropna(
            subset=['Obsah zmínek'])[
            ['id', 'Obsah zmínek']]
        vocabulary = list(pd.read_csv(
            WORKING_DIRECTORY + '/resources/word_vectorization_matrices/' + self.month + '/latest/occurrences_count_all.csv',
            names=['word_text', 'count']).loc[:5000, 'word_text'])
        vectorizer = CountVectorizer(vocabulary=vocabulary)
        return pd.DataFrame(index=id_to_mention['id'], columns=vocabulary,
                            data=vectorizer.transform(id_to_mention['Obsah zmínek']).todense())

    def add_relevance_tag(self):
        print('Adding relevance tag')
        id_to_relevance = \
        pd.read_csv(WORKING_DIRECTORY + '/resources/source_data/cleaner_data_' + self.month + '.csv').dropna(
            subset=['Obsah zmínek'])[['id', 'Štítek']].set_index('id')
        temp = id_to_relevance.applymap(func=lambda x: int(x == 'rel'))
        self.matrix = self.matrix.join(temp)
        return self

    def add_all_non_word_features(self):
        return self.add_author_features() \
            .add_domain_group_features() \
            .add_domain_features() \
            .add_mention_word_count_feature() \
            .add_mention_letter_count_feature() \
            .add_link_count_feature() \
            .add_mention_weekday_feature() \
            .add_mention_hour_feature() \
            .add_indicator_of_diacritics_usage()

def build_all_feature_matrices_for_given_month(filename):
    builder = FeatureMatrixBuilder(filename)
    month = filename.split('.')[0]

    print('building non word')
    builder.add_all_non_word_features().save(month + '_non_word')
    builder.reset_matrix()

    print('building word count')
    builder.add_word_count_features().save(month + '_word_count')
    builder.reset_matrix()

    print('building word indicator')
    builder.add_word_indicator_features().save(month + '_word_indicator')
    builder.reset_matrix()

#
# build_all_feature_matrices_for_given_month('prosinec')
# build_all_feature_matrices_for_given_month('rijen')

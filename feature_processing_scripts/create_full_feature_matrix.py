import pandas as pd
from datetime import datetime
import  re

class FeatureMatrixBuilder:

    def __init__(self, source):
        self.matrix = pd.read_csv(source).dropna(subset=['Obsah zmínek'])[['id', 'Datum vytvoření']].set_index('id')

    def build(self):
        return self.matrix

    def save(self):
        print('Saving...')
        regex = re.compile('[/ ]')
        time = regex.sub('_', datetime.now().strftime('%c').lower())
        self.matrix.to_csv('../resources/feature_matrices/feature_matrix_' + time + '.csv')


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


FeatureMatrixBuilder(source='../resources/general_data/cleaner_data.csv')\
    .add_domain_features()\
    .add_domain_group_features()\
    .add_author_features()\
    .save()

import pandas
import re
from general_data_scripts.load_data import get_replace_dictionary

dataset = pandas.read_csv('resources/general_data/cleaner_data.csv').dropna(subset=['Autor'])
Authors = dataset['Autor']


regex1 = re.compile('[^a-zA-Z0-9@]')
regex2 = re.compile('@[a-zA-Z0-9]*')
test0 = Authors.apply(func=lambda x: )
test1 = Authors.apply(func=lambda x: regex1.sub('', x))
test2 = test1.apply(func=lambda x: regex2.sub('', x))

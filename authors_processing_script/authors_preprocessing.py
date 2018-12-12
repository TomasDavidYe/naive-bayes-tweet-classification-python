import pandas
dataset = pandas.read_csv('resources/general_data/cleaner_data.csv').dropna(subset=['Autor'])
Authors = dataset['Autor']
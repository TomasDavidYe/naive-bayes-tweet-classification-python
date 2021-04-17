# About
As a member of a small university team, I analyzed vast amounts of twitter feed data and tried to answer the question: “How does the Czech public view nuclear energy?". 
The idea was to provide this answer by analysing a dataset of internet posts extracted from the Czech internet via NLP methods.

In this repository, the code implementing a Naive Bayes predictor of post relevance can be found.  


# Problem Statement
TODO


# Project Structure
The main entry point of the project is the 'run_model.py' script. Running this 


# Input Data

## Internet Posts
- **Description** -> Each row in this data represents an internet post with some metadata (author, platform, num likes...) with a tag of relevancy (1 if relevant, 0 if not). The goal is to train a model which will be able to categorize this data 
- **Example File** -> resources/source_data/cleare_data_rijen_prosinec.csv

## Stop Words
- **Description** -> Each row in this dataset contains a common Czech word which does not contribute to the meaning of a post and should be ignored by our model
- **Example File** -> resources/source_data/stop_words_13_12_2018.xlsx

# Accuracy
TODO
![](https://i.imgur.com/SWkz83x.png) 


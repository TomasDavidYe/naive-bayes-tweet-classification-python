# About
This package contains code which applies [**SciKit Learn's NaiveBayes**](https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes) to internet posts into 2 categories.
I built this package as a member of a university team when was analyzing twitter posts and tried to answer the question: “How do Czech people view nuclear energy?". 

I encapsulated the main logic into the [**NaiveBayes**](./model/NaiveBayes.py) class and this can be used to solve any similar text classification problem.
The original data is attached so you can quickly run an experiment yourself or you can feed it with your own data.

# Input Data

### 1) Internet Posts (example [here](./data/posts.csv))
**Columns:**
- id -> could be anything
- text -> string representing 
- is_relevant -> binary 0-1 variable indicating whether the corresponding text is relevant 


### 2) Stop Words (example [here](./data/stop_words.csv))
list of words that do not add value to classification and, hence, should be ignored by the classifier

# Installation
```bash
git clone https://github.com/TomasDavidYe/naive-bayes-tweet-classification-python;
cd naive-bayes-tweet-classification-python;
pip3 install -r requirements.txt;
```

# How to Run
To rerun the original experiment
```bash
python3 ./run_full_analysis.py 
```

To run on your data, make sure to supply the **posts.csv** and **stop_words.csv** data sets adhering to the schema described above and run
```python
from model.NaiveBayes import NaiveBayes

NaiveBayes(
    posts=posts, # Your posts
    stop_words=stop_words # Your stop words
).run_analysis(
    num_of_folds=2,
    classification_threshold=0.65,
    max_word_features=50
)
```  

# Meta Parameter Tunning
From plotting the ROC curve on multiple examples, it became clear that the default decision threshold of 0.5 is not optimal.  
![](https://i.imgur.com/3ZvUsOv.png)

By running different experiments on multiple stratified datasets and have found that a good value of the decision threshold for the specific Czech problem is **0.6**

### Default Threshold of 0.5
![](https://i.imgur.com/XfN2KOT.png)

### Experimentally verified Threshold of 0.6
![](https://i.imgur.com/AVXMFTj.png)
  



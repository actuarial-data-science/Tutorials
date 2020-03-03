#!/usr/bin/env python
# coding: utf-8

# # *The Art of Natural Language Processing: Machine Learning for the Case Study*

# ### **Authors: Andrea Ferrario, Mara Nägelin**

# **Date: February 2020**

# Notebook to run the machine learning modeling in the Classical and Modern Approaches, as described in the tutorial `The Art of Natural Language Processing: Classical, Modern and Contemporary Approaches to Text Document Classification'.

# # Table of contents
# 1. [Getting started with Python and Jupyter Notebook](#started)
# 2. [Import data](#import)
# 3. [Duplicated reviews](#duplicated)
# 4. [Data preprocessing](#preprocessing)
# 5. [POS-tagging](#POS)
# 6. [Pre-trained word embeddings](#emb)
# 7. [Data analytics](#analytics)  
#     7.1. [A quick check of data structure](#check)  
#     7.2. [Basic linguistic analysis of movie reviews](#basic)
# 8. [Machine learning](#ML)  
#     8.1. [Adaptive boosting (ADA)](#ADA)  
#     .......8.1.1. [Bag-of-words](#ADA_BOW)  
#     .......8.1.2. [Bag-of-POS](#ADA_BOP)  
#     .......8.1.3. [Embeddings](#ADA_E)  
#     8.2. [Random forests (RF)](#RF)  
#     .......8.2.1. [Bag-of-words](#RF_BOW)  
#     .......8.2.2. [Bag-of-POS](#RF_BOP)  
#     .......8.2.3. [Embeddings](#RF_E)  
#     8.3. [Extreme gradient boosting (XGB)](#XGB)  
#     .......8.3.1. [Bag-of-words](#XGB_BOW)  
#     .......8.3.2. [Bag-of-POS](#XGB_BOP)  
#     .......8.3.3. [Embeddings](#XGB_E)

# # 1. Getting started with Python and Jupyter Notebook<a name="started"></a>

# In this section, Jupyter Notebook and Python settings are initialized. For code in Python, the [PEP8 standard](https://www.python.org/dev/peps/pep-0008/) ("PEP = Python Enhancement Proposal") is enforced with minor variations to improve readibility.

# In[ ]:


# Notebook settings
###################

# resetting variables
get_ipython().magic('reset -sf') 

# formatting: cell width
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# plotting
get_ipython().run_line_magic('matplotlib', 'inline')


# # 2. Import data<a name="import"></a>

# In[ ]:


# we use the import function, as in Chapter 8 of Raschka's book (see the tutorial)
import pyprind
import pandas as pd
import os
basepath = '...' # insert basepath, where original data are stored

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 
                      'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], 
                           ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']


# # 3. Duplicated reviews<a name="duplicated"></a>

# In[ ]:


# check for duplicates
duplicates = df[df.duplicated()]  #equivalent to keep = first. Duplicated rows, except the first entry, are marked as 'True'
print(len(duplicates))


# In[ ]:


# a check on the duplicated review
duplicates.review   


# In[ ]:


# remove duplicates: 
df = df.drop_duplicates()
df.shape


# In[ ]:


# double check
df[df.duplicated(subset='review')]


# # 4. Data preprocessing<a name="preprocessing"></a>

# In[ ]:


# an example of 'raw' review: we have all sort of HTML markup
df.loc[500, 'review']


# In[ ]:


# preprocessing by Raschka, Chpater 8 (see tutorial)
# we remove all markups, substitute non-alphanumeric characters (including 
# underscore) with whitespaces, and remove the nose from emoticons
import re

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

df['review'] = df['review'].apply(preprocessor)


# In[ ]:


# checking again the same review
df.loc[500, 'review']


# In[ ]:


# save preprocessed data as csv 
path = '...'  # insert path
df.to_csv(path, index=False, encoding='utf-8')


# # 5. POS - tagging<a name="POS"></a>

# In[ ]:


# we apply POS-tagging on (deduplicated and) pre-processed data - let us import them
path = '...' # insert path
df = pd.read_csv(path, encoding='utf-8')
df.shape


# In[ ]:


# we import the NLTK resources
import nltk
from nltk import pos_tag, word_tokenize

# introduction of POS tagger per NLTK token
def pos_tags(text):
    text_processed = word_tokenize(text)
    return "-".join( tag for (word, tag) in nltk.pos_tag(text_processed))

# applying POS tagger to data 
############################################
df['text_pos']=df.apply(lambda x: pos_tags(x['review']), axis=1)


# In[ ]:


# save POS-tagged data as csv 
path = '...' # insert path
df.to_csv(path, index=False, encoding='utf-8')


# # 6. Pre-trained word embeddings<a name="emb"></a>

# In[ ]:


# we apply embeddings on de-duplicated and pre-processed data - let us import them
path = '...' # insert path
df = pd.read_csv(path, encoding='utf-8')
df.shape


# In[ ]:


# import pre-trained word embedding model
import spacy
nlp = spacy.load('en_core_web_md')


# In[ ]:


# we stack (like a numpy vertical stack) the 300 variables obtained from averaging the embedding of each df.review entry
import numpy as np
emb = np.vstack(df.review.apply(lambda x: nlp(x).vector))


# In[ ]:


# embeddings into a dataframe
emb = pd.DataFrame(emb, columns = np.array([str(x) for x in range(0, 299 + 1)]) )
print(emb.shape)
print(emb.columns.values)


# In[ ]:


# join embeddings with dataframe
df_embed = pd.concat([df, emb], axis=1)


# In[ ]:


# check the shape of the resulting dataframe
df_embed.shape


# In[ ]:


# save word embedding data as csv 
path = '...' # insert path
df_embed.to_csv(path, index=False, encoding='utf-8')


# # 7. Data analytics<a name="analytics"></a>

# We reproduce main data analytics results in Section 6.3 of the tutorial. We use the preprocessed and deduplicated data, for simplicity.

# ## 7.1. A quick check of data structure<a name="check"></a>

# In[ ]:


# importing data
import pandas as pd

path = '...' # insert path for deduplicated and preprocessed data
df = pd.read_csv(path)


# In[ ]:


# imported data structure
df.shape


# In[ ]:


# columns in data
df.columns


# In[ ]:


# imported data: first 10 entries
df.head(10)


# In[ ]:


# counts of rviews per sentiment value
df.sentiment.value_counts()


# ## 7.2. Basic linguistic analysis of movie reviews<a name="basic"></a>

# In[ ]:


# show distribution of review lenghts 
# we strip leading and trailing whitespaces and tokenize by whitespace
df['word_count'] = df['review'].apply(lambda x: len(x.strip().split(" ")))
df[['review','sentiment', 'word_count']].head(10)


# In[ ]:


# summary statistics of word counts
print(df['word_count'].describe())


# In[ ]:


# show histograms of word counts divided per sentiment
from matplotlib import pyplot

x = df[df['sentiment']==0].word_count
y = df[df['sentiment']==1].word_count

pyplot.hist(x, bins=50, alpha=0.5, label='negative sentiment reviews')
pyplot.hist(y, bins=50, alpha=0.5, label='positive sentiment reviews')
pyplot.legend(loc='upper right')
pyplot.show()


# In[ ]:


# summary of distributions of word counts
print(x.describe())
print(y.describe())


# In[ ]:


# some checks (e.g. word_counts=6 or 1550 or 2498 )
df[df['word_count']==6]


# In[ ]:


# average word length (again, we tokenize by whitespaces)
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

df['avg_word'] = df['review'].apply(lambda x: avg_word(x.strip()))
df[['review','word_count', 'sentiment', 'avg_word']].head(10)


# In[ ]:


# distributions of word lengths conditional per sentiment
x = df[df['sentiment']==0].avg_word
y = df[df['sentiment']==1].avg_word
print(x.describe())
print()
print(y.describe())


# In[ ]:


# some checks (e.g. avg_word>=11)
df[df['avg_word']>=11]


# In[ ]:


# stop words statistics - stopword from NLTK
from nltk.corpus import stopwords
stop = stopwords.words('english')

df['stopwords'] = df['review'].apply(lambda x: len([x for x in x.strip().split() if x in stop]))
df[['review','word_count', 'sentiment', 'avg_word', 'stopwords']].head(10)


# In[ ]:


# distributions of stop words conditional per sentiment
x = df[df['sentiment']==0].stopwords
y = df[df['sentiment']==1].stopwords
print(x.describe())
print()
print(y.describe())


# In[ ]:


# some checks (e.g. stopwords==0)
df[df['stopwords']==0]


# # 8. Machine Learning<a name="ML"></a>

# We replicate the machine learning pipelines from the tutorial, Section 6.4 (Classical and Modern Approaches).

# **WARNING**: as mentioned in the tutorial, the following cross-validation routines are computationally intensive. We recommend to sub-sample data and/or use HPC infrastructure (specifying the parameter njobs in GridSearch() accordingly). Test runs can be launched on reduced hyperparameter grids, as well.

# ## 8.1. Adaptive boosting (ADA)<a name="ADA"></a>

# We use the adaptive boosting (ADA) algorithm on top of NLP pipelines (bag-of models and pre-trained word embeddings).

# ### 8.1.1. Bag-of-words<a name="ADA_BOW"></a>

# In[ ]:


# loading Python packages
#########################

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve
from sklearn.metrics import auc 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# data preparation
###########################################

# data import 
import pandas as pd

path = '...'           # insert path to deduplicated and preprocessed data
df = pd.read_csv(path)     

# shuffling data
import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

# introducing the stopwords
###########################################

# stopwords
import nltk
from nltk.corpus import stopwords
stopwords = list(set(stopwords.words('english')))

# train vs. test: we already shuffled data -80/20 split
##########################################

X_train = df.head(39666).review
y_train = df.head(39666).sentiment

X_test = df.tail(9916).review
y_test = df.tail(9916).sentiment

print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('X_train shape check: ', X_train.shape)
print('X_test shape check: ', X_test.shape)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')

# GridSearch()
###########################################

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = {'vect__ngram_range': [(1, 1)],           # choose (1, 2) to compute 2-grams
              'vect__stop_words': [stopwords, None],
              'vect__max_df': [1.0, 0.1, 0.3, 0.5],
              'vect__max_features': [None, 1000],                                          
              'clf__n_estimators': [100, 200, 300, 400],
              'clf__learning_rate': [0.001, 0.01, 0.1, 1.0]
              }

tree = DecisionTreeClassifier(max_depth=5)

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', AdaBoostClassifier(base_estimator=tree))]
                    )

# on cross-validation parameters
cv = StratifiedKFold(n_splits=5, 
                     shuffle=False
                     )

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='roc_auc',
                           cv=cv, 
                           n_jobs=)               # insert the number of jobs, according to the used machine

# running the grid
###########################################

gs_lr_tfidf.fit(X_train, y_train)

# best estimator - test performance
###########################################

clf_b = gs_lr_tfidf.best_estimator_
y_pred_proba = clf_b.predict_proba(X_test)
y_pred = clf_b.predict(X_test)

# AUC on test data
auc_res=roc_auc_score(y_test, y_pred_proba[:, 1])

# Accuracy on test data
acc = accuracy_score(y_test, y_pred)

# collecting results
###########################################

print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('Test AUC: %.3f' % auc_res)
print('Test Accuracy: %.3f' % acc)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')


# ### 8.1.2. Bag-of-POS<a name="ADA_BOP"></a>

# In[ ]:


# loading Python packages
#########################

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve
from sklearn.metrics import auc 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# data preparation
###########################################

# data import
import pandas as pd

path = '...'  # insert path to data with POS-tags
df = pd.read_csv(path)

# shuffling data
import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

# train vs. test: we already shuffled data -80/20 split
##########################################

X_train = df.head(39666).text_pos
y_train = df.head(39666).sentiment

X_test = df.tail(9916).text_pos
y_test = df.tail(9916).sentiment

print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('X_train shape check: ', X_train.shape)
print('X_test shape check: ', X_test.shape)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')

# GridSearch()
###########################################

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = {'vect__ngram_range': [(1, 1)],               # we consider only 1-gram POS (for 2-grams: (1,2))
              'clf__n_estimators': [100, 200, 300, 400],
              'clf__learning_rate': [0.001, 0.01, 0.1, 1.0]}

tree = DecisionTreeClassifier(max_depth=5)

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', AdaBoostClassifier(base_estimator=tree))]
                    )

# on cross-validation parameters
cv = StratifiedKFold(n_splits=5, 
                     shuffle=False
                     )

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='roc_auc',
                           cv=cv, 
                           n_jobs=)                        # insert the number of jobs, according to the used machine

# running the grid
###########################################

gs_lr_tfidf.fit(X_train, y_train)

# best estimator - test performance
###########################################

clf_b = gs_lr_tfidf.best_estimator_
y_pred_proba = clf_b.predict_proba(X_test)
y_pred = clf_b.predict(X_test)

# AUC on test data
auc_res=roc_auc_score(y_test, y_pred_proba[:, 1])

# Accuracy on test data
acc = accuracy_score(y_test, y_pred)

# collecting results
###########################################

print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('Test AUC: %.3f' % auc_res)
print('Test Accuracy: %.3f' % acc)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')


# ### 8.1.3. Embeddings<a name="ADA_E"></a>

# In[ ]:


# loading Python packages
#########################

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve
from sklearn.metrics import auc 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# data preparation
###########################################

# data import
import pandas as pd

path = '...'  # insert path to data with pre-trained word embeddings
df = pd.read_csv(path)

# shuffling data
import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

# train vs. test: we already shuffled data -80/20 split - drop the variables
############################################################################

X_train = df.drop(columns=['review', 'sentiment']).head(39666)    # we use only the 300 embeddings
y_train = df.head(39666).sentiment

X_test = df.drop(columns=['review', 'sentiment']).tail(9916)      # we use only the 300 embeddings
y_test = df.tail(9916).sentiment

print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('X_train shape check: ', X_train.shape)
print('X_test shape check: ', X_test.shape)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')

# GridSearch()
###########################################

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {'clf__n_estimators': [100, 200, 300, 400],
              'clf__learning_rate': [0.001, 0.01, 0.1, 1.0]}

# extended parameter grid (Table 6, Section 6.4.5 in the tutorial)
# param_grid = {'clf__n_estimators': [100, 200, 300, 400, 500, 700, 900, 1000],
#              'clf__learning_rate': [0.001, 0.01, 0.1, 1.0]}

tree = DecisionTreeClassifier(max_depth=5)

pipe = Pipeline([('clf', AdaBoostClassifier(base_estimator=tree))])

# on cross-validation parameters
cv = StratifiedKFold(n_splits=5, 
                     shuffle=False
                     )

gs_lr_tfidf = GridSearchCV(pipe, param_grid,
                           scoring='accuracy',
                           cv=cv, 
                           n_jobs=)                        # insert the number of jobs, according to the used machine

# running the grid
###########################################

gs_lr_tfidf.fit(X_train, y_train)

# best estimator - test performance
###########################################

clf_b = gs_lr_tfidf.best_estimator_
y_pred_proba = clf_b.predict_proba(X_test)
y_pred = clf_b.predict(X_test)

# AUC on test data
auc_res=roc_auc_score(y_test, y_pred_proba[:, 1])

# Accuracy on test data
acc = accuracy_score(y_test, y_pred)

# collecting results
###########################################

print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('Test AUC: %.3f' % auc_res)
print('Test Accuracy: %.3f' % acc)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')


# ## 8.2. Random Forests (RF)<a name="RF"></a>

# We use the random forests (RF) algorithm on top of NLP pipelines (bag-of models and pre-trained word embeddings).

# ### 8.2.1. Bag-of-words<a name="RF_BOW"></a>

# In[ ]:


# loading Python packages
#########################

from sklearn.utils import shuffle 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve
from sklearn.metrics import auc 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# data preparation
###########################################

# data import
import pandas as pd

path = '...' # insert path to deduplicated and preprocessed data
df = pd.read_csv(path)

# shuffling data
import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

# introducing the stopwords
###########################################

# stopwords
import nltk
from nltk.corpus import stopwords
stopwords = list(set(stopwords.words('english')))

# train vs. test: we already shuffled data -80/20 split
##########################################

X_train = df.head(39666).review
y_train = df.head(39666).sentiment

X_test = df.tail(9916).review
y_test = df.tail(9916).sentiment

print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('X_train shape check: ', X_train.shape)
print('X_test shape check: ', X_test.shape)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')

# GridSearch()
###########################################

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = {'vect__ngram_range': [(1, 1)],    # for 2-grams:(1, 2)
              'vect__stop_words': [stopwords, None],
              'vect__max_df': [1.0, 0.1, 0.3, 0.5],
              'vect__max_features': [None, 1000],                                           
              'clf__n_estimators': [100, 200, 300, 400],
              'clf__max_depth': [1, 5, 10]
              }

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', RandomForestClassifier())]
                    )

# on cross-validation parameters
cv = StratifiedKFold(n_splits=5, 
                     shuffle=False
                     )

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='roc_auc',
                           cv=cv, 
                           n_jobs=)       # insert the number of jobs, according to the used machine

# running the grid
###########################################

gs_lr_tfidf.fit(X_train, y_train)

# best estimator - test performance
###########################################

clf_b = gs_lr_tfidf.best_estimator_
y_pred_proba = clf_b.predict_proba(X_test)
y_pred = clf_b.predict(X_test)

# AUC on test data
auc_res=roc_auc_score(y_test, y_pred_proba[:, 1])

# Accuracy on test data
acc = accuracy_score(y_test, y_pred)

# collecting results
###########################################

print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('Test AUC: %.3f' % auc_res)
print('Test Accuracy: %.3f' % acc)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')


# ### 8.2.2. Bag-of-POS<a name="RF_BOP"></a>

# In[ ]:


# loading Python packages
#########################

from sklearn.utils import shuffle 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve
from sklearn.metrics import auc 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# data preparation
###########################################

# data import
import pandas as pd

path = '...'  # insert path to data with POS-tags
df = pd.read_csv(path)

# shuffling data
import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

# train vs. test: we already shuffled data -80/20 split
##########################################

X_train = df.head(39666).text_pos
y_train = df.head(39666).sentiment

X_test = df.tail(9916).text_pos
y_test = df.tail(9916).sentiment

print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('X_train shape check: ', X_train.shape)
print('X_test shape check: ', X_test.shape)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')

# GridSearch()
###########################################

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = {'vect__ngram_range': [(1, 1)],               # we consider only 1-gram POS (for 2-grams: (1,2))
              'clf__n_estimators': [100, 200, 300, 400],
              'clf__max_depth': [1, 5, 10]
              }

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', RandomForestClassifier())]
                    )

# on cross-validation parameters
cv = StratifiedKFold(n_splits=5, 
                     shuffle=False
                     )

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='roc_auc',
                           cv=cv, 
                           n_jobs=)              # insert the number of jobs, according to the used machine

# running the grid
###########################################

gs_lr_tfidf.fit(X_train, y_train)

# best estimator - test performance
###########################################

clf_b = gs_lr_tfidf.best_estimator_
y_pred_proba = clf_b.predict_proba(X_test)
y_pred = clf_b.predict(X_test)

# AUC on test data
auc_res=roc_auc_score(y_test, y_pred_proba[:, 1])

# Accuracy on test data
acc = accuracy_score(y_test, y_pred)

# collecting results
###########################################

print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('Test AUC: %.3f' % auc_res)
print('Test Accuracy: %.3f' % acc)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')


# ### 8.2.3. Embeddings<a name="RF_E"></a>

# In[ ]:


# loading Python packages
#########################

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve
from sklearn.metrics import auc 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# data preparation
###########################################

# data import 
import pandas as pd

path = '...'  # insert path to data with pre-trained word embeddings
df = pd.read_csv(path)

# shuffling data
import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

# train vs. test: we already shuffled data -80/20 split - drop the variables
############################################################################

X_train = df.drop(columns=['review', 'sentiment']).head(39666)    # we use only the 300 embeddings
y_train = df.head(39666).sentiment

X_test = df.drop(columns=['review', 'sentiment']).tail(9916)     # we use only the 300 embeddings
y_test = df.tail(9916).sentiment

print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('X_train shape check: ', X_train.shape)
print('X_test shape check: ', X_test.shape)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')

# GridSearch()
###########################################

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


param_grid = {
              'clf__n_estimators': [100, 200, 300, 400],
              'clf__max_depth': [1, 5, 10]
             }

# extended parameter grid (Table 6, Section 6.4.5 in the tutorial)
# param_grid = {'clf__n_estimators': [100, 200, 300, 400, 500, 600, 800, 1000],
#              'clf__max_depth': [1, 5, 10, 20]}

pipe = Pipeline([('clf', RandomForestClassifier())
               ])

# on cross-validation parameters
cv = StratifiedKFold(n_splits=5, 
                     shuffle=False
                     )

gs_lr_tfidf = GridSearchCV(pipe, param_grid,
                           scoring='roc_auc',
                           cv=cv, 
                           n_jobs=)               # insert the number of jobs, according to the used machine

# running the grid
###########################################

gs_lr_tfidf.fit(X_train, y_train)

# best estimator - test performance
###########################################

clf_b = gs_lr_tfidf.best_estimator_
y_pred_proba = clf_b.predict_proba(X_test)
y_pred = clf_b.predict(X_test)

# AUC on test data
auc_res=roc_auc_score(y_test, y_pred_proba[:, 1])

# Accuracy on test data
acc = accuracy_score(y_test, y_pred)

# collecting results
###########################################

print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('Test AUC: %.3f' % auc_res)
print('Test Accuracy: %.3f' % acc)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')


# ## 8.3. Extreme gradient boosting (XGB)<a name="XGB"></a>

# We use the extreme gradient boosting (XGB) algorithm on top of NLP pipelines (bag-of models and pre-trained word embeddings). We can use the cell below to install xgboost, if other imports failed.

# In[ ]:


# importing xgboost REMARK: run this cell only if other imports failed. Delete it in case xgboost has been already imported
import pip
pip.main(['install', 'xgboost'])


# ### 8.3.1. Bag-of-words<a name="XGB_BOW"></a>

# In[ ]:


# loading Python packages
#########################

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve
from sklearn.metrics import auc 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# data preparation
###########################################

# data import
import pandas as pd

path = '...'  # insert path to preprocessed and deduplicated data
df = pd.read_csv(path)

# shuffling data
import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

# introducing the stopwords
###########################################

# stopwords
import nltk
from nltk.corpus import stopwords
stopwords = list(set(stopwords.words('english')))

# train vs. test: we already shuffled data -80/20 split
##########################################

X_train = df.head(39666).review
y_train = df.head(39666).sentiment

X_test = df.tail(9916).review
y_test = df.tail(9916).sentiment

print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('X_train shape check: ', X_train.shape)
print('X_test shape check: ', X_test.shape)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')

# GridSearch()
###########################################

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = {'vect__ngram_range': [(1, 1)]    # for 2-grams: (1,2)
              'vect__stop_words': [stopwords, None],
              'vect__max_df': [1.0, 0.1, 0.3, 0.5],
              'vect__max_features': [None, 1000],                                           
              'clf__n_estimators': [100, 300, 500, 1000],
              'clf__learning_rate': [0.001, 0.01, 0.1, 1.0],
              'clf__max_depth': [1, 10, 20]
              }

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', XGBClassifier())]
                    )

# on cross-validation parameters
cv = StratifiedKFold(n_splits=5, 
                     shuffle=False
                     )

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='roc_auc',
                           cv=cv, 
                           n_jobs=)                 # insert the number of jobs, according to the used machine

# running the grid
###########################################

gs_lr_tfidf.fit(X_train, y_train)

# best estimator - test performance
###########################################

clf_b = gs_lr_tfidf.best_estimator_
y_pred_proba = clf_b.predict_proba(X_test)
y_pred = clf_b.predict(X_test)

# AUC on test data
auc_res=roc_auc_score(y_test, y_pred_proba[:, 1])

# Accuracy on test data
acc = accuracy_score(y_test, y_pred)

# collecting results
###########################################

print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('Test AUC: %.3f' % auc_res)
print('Test Accuracy: %.3f' % acc)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')


# ### 8.3.2. Bag-of-POS<a name="XGB_BOP"></a>

# In[ ]:


# loading Python packages
#########################

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve
from sklearn.metrics import auc 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# data preparation
###########################################

# data import
import pandas as pd

path = '...'  # insert data with POS-tags
df = pd.read_csv(path)

# shuffling data
import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

# train vs. test: we already shuffled data -80/20 split
##########################################

X_train = df.head(39666).review
y_train = df.head(39666).sentiment

X_test = df.tail(9916).review
y_test = df.tail(9916).sentiment

print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('X_train shape check: ', X_train.shape)
print('X_test shape check: ', X_test.shape)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')

# GridSearch()
###########################################

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = {'vect__ngram_range': [(1, 1)],               # we consider only 1-gram POS (for 2-grams: (1,2))
              'clf__n_estimators': [100, 300, 500, 1000],
              'clf__learning_rate': [0.001, 0.01, 0.1, 1.0],
              'clf__max_depth': [1, 10, 20]
              }

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', XGBClassifier())]
                    )


# on cross-validation parameters
cv = StratifiedKFold(n_splits=5, 
                     shuffle=False
                     )


gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='roc_auc',
                           cv=cv, 
                           n_jobs=)              # insert the number of jobs, according to the used machine

# running the grid
###########################################

gs_lr_tfidf.fit(X_train, y_train)

# best estimator - test performance
###########################################

clf_b = gs_lr_tfidf.best_estimator_
y_pred_proba = clf_b.predict_proba(X_test)
y_pred = clf_b.predict(X_test)

# AUC on test data
auc_res=roc_auc_score(y_test, y_pred_proba[:, 1])

# Accuracy on test data
acc = accuracy_score(y_test, y_pred)

# collecting results
###########################################

print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('Test AUC: %.3f' % auc_res)
print('Test Accuracy: %.3f' % acc)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')


# ### 8.3.3. Embeddings<a name="XGB_E"></a>

# In[ ]:


# loading Python packages
#########################

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve
from sklearn.metrics import auc 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# data preparation
###########################################

# data import
import pandas as pd

path = '...'  # insert data with pre-trained word embeddings
df = pd.read_csv(path)

# shuffling data
import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

# train vs. test: we already shuffled data -80/20 split - drop the variables
############################################################################

X_train = df.drop(columns=['review', 'sentiment']).head(39666)    # we use only the 300 embeddings
y_train = df.head(39666).sentiment

X_test = df.drop(columns=['review', 'sentiment']).tail(9916)     # we use only the 300 embeddings
y_test = df.tail(9916).sentiment

print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('X_train shape check: ', X_train.shape)
print('X_test shape check: ', X_test.shape)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')

# GridSearch()
###########################################

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
              'clf__n_estimators': [100, 300, 500, 1000],
              'clf__learning_rate': [0.001, 0.01, 0.1, 1.0],
              'clf__max_depth': [1, 10, 20]
              }

pipe = Pipeline([('clf', XGBClassifier())
                ])
                    
# on cross-validation parameters
cv = StratifiedKFold(n_splits=5, 
                     shuffle=False
                     )

gs_lr_tfidf = GridSearchCV(pipe, param_grid,
                           scoring='roc_auc',
                           cv=cv, 
                           n_jobs=)        # insert the number of jobs, according to the used machine

# running the grid
###########################################

gs_lr_tfidf.fit(X_train, y_train)

# best estimator - test performance
###########################################

clf_b = gs_lr_tfidf.best_estimator_
y_pred_proba = clf_b.predict_proba(X_test)
y_pred = clf_b.predict(X_test)

# AUC on test data
auc_res=roc_auc_score(y_test, y_pred_proba[:, 1])

# Accuracy on test data
acc = accuracy_score(y_test, y_pred)

# collecting results
###########################################

print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')
print('Test AUC: %.3f' % auc_res)
print('Test Accuracy: %.3f' % acc)
print('---------------------')
print('---------------------')
print('---------------------')
print('---------------------')


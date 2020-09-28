# *The Art of Natural Language Processing: RNNs for the Case Study*

### **Authors: Andrea Ferrario, Mara Nägelin**

**Date: February 2020** (updated September 2020)

Notebook to run the RNNs in the Contemporary Approach, as described in the tutorial `The Art of Natural Language Processing: Classical, Modern and Contemporary Approaches to Text Document Classification'.

# Table of contents
1. [Getting started with Python and Jupyter Notebook](#datagen)
2. [Import data](#datagen)
3. [Data preprocessing](#dataprep)  
    3.1. [Remove duplicates](#remdup)  
    3.2. [Shuffle the data](#shuffle)  
    3.3. [Minimal preprocessing (detailed)](#prep_det)  
    3.4. [Minimal preprocessing with Keras](#prep_keras)  
4. [Deep learning](#DL)  
    4.1. [Train test split](#trainsplit)  
    4.2. [Define the model](#modeldef)  
    .......4.2.1. [Shallow LSTM](#LSTM1)  
    .......4.2.2. [Shallow GRU](#GRU1)  
    .......4.2.3. [Deep LSTM](#LSTM2)    
    4.3. [Train the model](#trainmodel)  
    4.4. [Evaluate the model on test data](#evalmodel)  
5. [Final remarks](#fm)

<a name="started"></a>
# 1. Getting started with Python and Jupyter Notebook

In this section, Jupyter Notebook and Python settings are initialized. For code in Python, the [PEP8 standard](https://www.python.org/dev/peps/pep-0008/) ("PEP = Python Enhancement Proposal") is enforced with minor variations to improve readibility.


```python
# Notebook settings
###################

# resetting variables
get_ipython().magic('reset -sf') 

# formatting: cell width
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# plotting
%matplotlib inline
```


<style>.container { width:100% !important; }</style>


<a id='datagen'></a>

# 2. Import data

First, we import the raw data from the original 50'000 text files and save to a dataframe. This only needs to be run once. After that, one can start directly with [Section 2](#dataprep). The following code snippet is based on the book `Python Machine Learning` by Raschka and Mirjalili, Chapter 8 (see tutorial).


```python
import pyprind
import pandas as pd
import os
basepath = 'path_to_extracted_data/aclImdb/' # TODO: update to point to your data repository

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
```

    0% [##############################] 100% | ETA: 00:00:00
    Total time elapsed: 00:01:46
    


```python
# save as csv 
path = "path_to_save_data/movie_data.csv" # TODO: update to your path
df.to_csv(path, index=False, encoding='utf-8')
```

<a id='dataprep'></a>

# 3. Data preprocessing

Next we preprare the raw data such that in can be used as input for a neural network. Again, we follow the example of Raschka and Mirjalili (Chapter 16). We perform the following steps:

* We remove all duplicates.
* We shuffle the data in a random permutation.
* We apply only minimal preprocessing (i.e. convert to lowercase and split on whitespaces and punctuation).
* We map each word bijectively to an integer value.
* We set each review to an equal length $T$ by padding with $0$ or slicing as required.

The last three steps are written out in detail in [Section 2.3.](#prep_det) to give the reader an understanding of what exactly happens to the data. However, they can also be carried out — almost equivalently — using the high-level `text.preprocessing` functionalities of the `tensorflow.keras` module, see [Section 2.4.](#prep_keras) The user needs to run only one of these two subsections to preprocess the data.

The transformed data is stored in a dataframe for convenience. Hence [Section 2](#dataprep) also needs to be run only once, and after one can start jump directly to [Section 3](#DL).

The following can be used to reimport the dataframe with the raw data generated in [Section 1](#datagen) above.


```python
# import the data
import pandas as pd
import os

path = 'path_to_save_data/movie_data.csv' # TODO: update to your path  
df = pd.read_csv(path, encoding='utf-8') # read in the dataframe stored as csv
df.shape
```




    (50000, 2)



<a id='remdup'></a>

## 3.1. Remove duplicates


```python
# check for duplicates - we found them, even with HTML markup...
duplicates = df[df.duplicated()]  # Duplicated rows, except the first entry, are marked as 'True'
print(len(duplicates))
```

    418
    


```python
# a check on the duplicated review
duplicates.review   # some appear more than once, as they originally appear 3 or more times in the dataset
```




    33       I was fortunate to attend the London premier o...
    177      I've been strangely attracted to this film sin...
    939      The Andrew Davies adaptation of the Sarah Wate...
    1861     <br /><br />First of all, I reviewed this docu...
    1870     Spheeris debut must be one of the best music d...
                                   ...                        
    49412    There is no way to avoid a comparison between ...
    49484    **SPOILERS** I rented "Tesis" (or "Thesis" in ...
    49842    'Dead Letter Office' is a low-budget film abou...
    49853    This movie had a IMDB rating of 8.1 so I expec...
    49864    You know all those letters to "Father Christma...
    Name: review, Length: 418, dtype: object




```python
# remove duplicates: 49582 + 418 = 50000
df = df.drop_duplicates()
df.shape
```




    (49582, 2)




```python
# double check
df[df.duplicated(subset='review')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



<a id='shuffle'></a>

## 3.2. Shuffle the data


```python
# We shuffle the data to ensure randomness in the training input
import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df = df.reset_index(drop=True) # reset the index after the permutation
```

<a id='prep_det'></a>

## 3.3. Minimal preprocessing (detailed)

The following snippets are in part adapted from Raschka and Mirjalili (Chapter 16).


```python
# Minimal preprocessing and generating word counts:
#  - we surround all punctuation by whitespaces
#  - all text is converted to lowercase
#  - word counts are generated by splitting the text on whitespaces
import pyprind
from string import punctuation
from collections import Counter

counts = Counter()
pbar = pyprind.ProgBar(len(df['review']), title='Counting words occurrences')
for i,review in enumerate(df['review']):
    text = ''.join([c if c not in punctuation else ' '+c+' '  
                    for c in review]).lower()
    df.loc[i,'review'] = text
    pbar.update()
    counts.update(text.split()) # splitting on whitespace
```

    Counting words occurrences
    0% [##############################] 100% | ETA: 00:00:00
    Total time elapsed: 00:07:16
    


```python
# get the size of the vocabulary
print('Number of unique words:', len(counts))
```

    Number of unique words: 102966
    


```python
# investigate how many words appear only rarely in the reviews
print('Number of words that appear more than once:', 
      len([k for k, v in counts.items() if v > 1]))
print('Number of words that appear more than 30 times:', 
      len([k for k, v in counts.items() if v > 30]))
```

    Number of words that appear more than once: 62923
    Number of words that appear more than 30 times: 15282
    


```python
# hence we use only the 15'000 most common in our vocabulary 
# this will make training more efficient without loosing too much information
vocab_size = 15000

# create a dictionary with word:integer pairs for all unique words
word_counts = sorted(counts, key=counts.get, reverse=True)
word_counts = word_counts[0:vocab_size]
word_to_int = {word: ii for ii, word in enumerate(word_counts, 1)}
```


```python
# Mapping words to integers
# create a list with all reviews in integer coded form
mapped_reviews = []
pbar = pyprind.ProgBar(len(df['review']), title='Map reviews to ints')
for review in df['review']:
    mapped_reviews.append([word_to_int[word] 
                           for word in review.split() 
                           if word in word_to_int.keys()])
    pbar.update()
```

    Map reviews to ints
    0% [##############################] 100% | ETA: 00:00:00
    Total time elapsed: 00:00:08
    


```python
# get the median length of the mapped review sequences to inform the choice of sequence_length
print('Median length of mapped reviews:',
      np.median([len(review) for review in mapped_reviews]))
```

    Median length of mapped reviews: 213.0
    


```python
import matplotlib.pyplot as plt
rev_lengths = np.array([len(review) for review in mapped_reviews])
plt.hist(rev_lengths[rev_lengths < 500])
plt.show()
```


    
![png](output_36_0.png)
    



```python
# Padding: set sequence length and ensure all mapped reviews are coerced to required length
# if sequence length < T: left-pad with zeros
# if sequence length > T: use the last T elements
sequence_length = 200  # (Known as T in our RNN formulae)
sequences = np.zeros((len(mapped_reviews), sequence_length), dtype=int)

for i, row in enumerate(mapped_reviews):
    review_arr = np.array(row)
    sequences[i, -len(row):] = review_arr[-sequence_length:]
```


```python
# create df with processed data
df_processed = pd.concat([df['sentiment'],pd.DataFrame(sequences)], axis=1)
df_processed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sentiment</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>190</th>
      <th>191</th>
      <th>192</th>
      <th>193</th>
      <th>194</th>
      <th>195</th>
      <th>196</th>
      <th>197</th>
      <th>198</th>
      <th>199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>961</td>
      <td>306</td>
      <td>581</td>
      <td>82</td>
      <td>97</td>
      <td>1</td>
      <td>76</td>
      <td>2</td>
      <td>...</td>
      <td>11</td>
      <td>1945</td>
      <td>1209</td>
      <td>2</td>
      <td>1881</td>
      <td>3</td>
      <td>2612</td>
      <td>3</td>
      <td>5244</td>
      <td>2218</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14</td>
      <td>8</td>
      <td>21</td>
      <td>147</td>
      <td>1074</td>
      <td>25</td>
      <td>463</td>
      <td>6240</td>
      <td>41</td>
      <td>41</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>359</td>
      <td>1</td>
      <td>109</td>
      <td>3</td>
      <td>44</td>
      <td>2053</td>
      <td>9122</td>
      <td>6</td>
      <td>...</td>
      <td>3996</td>
      <td>678</td>
      <td>105</td>
      <td>1908</td>
      <td>2667</td>
      <td>2</td>
      <td>5</td>
      <td>360</td>
      <td>24</td>
      <td>2</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>2728</td>
      <td>16</td>
      <td>4074</td>
      <td>40</td>
      <td>493</td>
      <td>7</td>
      <td>89</td>
      <td>14</td>
      <td>2</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>19</td>
      <td>1440</td>
      <td>19</td>
      <td>723</td>
      <td>19</td>
      <td>1855</td>
      <td>19</td>
      <td>1440</td>
      <td>19</td>
      <td>...</td>
      <td>1201</td>
      <td>19</td>
      <td>2294</td>
      <td>6281</td>
      <td>10</td>
      <td>64</td>
      <td>6112</td>
      <td>771</td>
      <td>100</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 201 columns</p>
</div>



<a id='prep_keras'></a>

## 3.4. Minimal preprocessing with Keras


```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 15000

# map words to integers including minimal preprocessing
tokenizer = Tokenizer(num_words=vocab_size, 
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', # filters out all punctuation other than '
                      lower=True, # convert to lowercase
                      split=' ') # split on whitespaces
tokenizer.fit_on_texts(df['review'])
list_tokenized = tokenizer.texts_to_sequences(df['review'])

# Padding to sequence_length
sequence_length = 200
sequences = pad_sequences(list_tokenized, maxlen=sequence_length)

# create df with processed data
df_processed = pd.concat([df['sentiment'], pd.DataFrame(sequences)], axis=1)
df_processed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sentiment</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>190</th>
      <th>191</th>
      <th>192</th>
      <th>193</th>
      <th>194</th>
      <th>195</th>
      <th>196</th>
      <th>197</th>
      <th>198</th>
      <th>199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1786</td>
      <td>1003</td>
      <td>1427</td>
      <td>3</td>
      <td>832</td>
      <td>7</td>
      <td>20</td>
      <td>48</td>
      <td>3</td>
      <td>...</td>
      <td>20</td>
      <td>1132</td>
      <td>8</td>
      <td>8</td>
      <td>1923</td>
      <td>1192</td>
      <td>1859</td>
      <td>2590</td>
      <td>5216</td>
      <td>2194</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>23</td>
      <td>263</td>
      <td>9</td>
      <td>6</td>
      <td>14</td>
      <td>133</td>
      <td>1057</td>
      <td>18</td>
      <td>448</td>
      <td>6213</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>10587</td>
      <td>27</td>
      <td>3</td>
      <td>1133</td>
      <td>1637</td>
      <td>374</td>
      <td>26</td>
      <td>...</td>
      <td>12</td>
      <td>997</td>
      <td>3971</td>
      <td>663</td>
      <td>93</td>
      <td>1886</td>
      <td>2643</td>
      <td>3</td>
      <td>345</td>
      <td>17</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>1</td>
      <td>2706</td>
      <td>11</td>
      <td>4049</td>
      <td>31</td>
      <td>478</td>
      <td>5</td>
      <td>77</td>
      <td>9</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>10588</td>
      <td>3559</td>
      <td>461</td>
      <td>4106</td>
      <td>14422</td>
      <td>116</td>
      <td>2</td>
      <td>3836</td>
      <td>560</td>
      <td>...</td>
      <td>117</td>
      <td>13</td>
      <td>1183</td>
      <td>2271</td>
      <td>6295</td>
      <td>7</td>
      <td>53</td>
      <td>6085</td>
      <td>755</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 201 columns</p>
</div>



<br>  
<br>
Lastly, we save the fully preprocesssed data to csv for further use.


```python
# save as csv 
path = "path_to_save_data/movie_data_processed.csv" # TODO: update to your path
df_processed.to_csv(path, index=False, encoding='utf-8')
```

<a id='DL'></a>

# 4. Deep Learning

In this section, we reproduce the results from Section 6.4.6-6.4.7 of the tutorial. We split the preprocessed data into a training and a testing set and define our RNN model (three different possible architectures are given as examples, see [Sections 4.2.1.-4.2.3](#LSTM1). The model is compiled and trained using the high-level `tensorflow.keras` API. Finally, the development of loss and accuracy during the training is plotted and the fitted model is evaluated on the test data.  

**WARNING:** Note that training with a large training dataset for a large number of epochs is computationally intensive and might easily take a couple of hours on a normal CPU machine. We recommend subsetting the training and testing datasets and/or using an HPC infrastructure.


```python
# to ensure that all keras functionalities work as intended
from __future__ import absolute_import, division, print_function, unicode_literals
```


```python
# import the data
import pandas as pd
import os
import numpy as np

path = "path_to_save_data/movie_data_processed.csv" # TODO: update to your path
df_processed = pd.read_csv(path, encoding='utf-8')
df_processed.shape
```




    (49582, 201)



<a id='trainsplit'></a>

## 4.1. Train test split


```python
# get the number of samples for the training and test datasets
perc_train = 0.8
n_train = round(df_processed.shape[0]*perc_train)
n_test = round(df_processed.shape[0]*(1-perc_train))

print(str(int(perc_train*100))+'/'+str(int(100-perc_train*100))+' train test split:', 
      n_train, '/', n_test)
```

    80/20 train test split: 39666 / 9916
    


```python
# create the training and testing datasets
X_train = np.array(df_processed.head(n_train).drop('sentiment', axis=1)) # replace with n_train
y_train = df_processed.head(n_train).sentiment.values

X_test = np.array(df_processed.tail(n_test).drop('sentiment', axis=1)) # replace with n_test
y_test = df_processed.tail(n_test).sentiment.values

print('Training data shape check X, y:', X_train.shape, y_train.shape)
print('Testing data shape check X, y:', X_test.shape, y_test.shape)
```

    Training data shape check X, y: (39666, 200) (39666,)
    Testing data shape check X, y: (9916, 200) (9916,)
    

<a id='modeldef'></a>

## 4.2. Define the model


```python
import collections
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

# get the total number of words in vocabulary (+1 for the padding with 0)
vocab_size = df_processed.drop('sentiment', axis=1).values.max() + 1
print('Vocabulary size:', vocab_size-1)
```

    Vocabulary size: 14999
    

Each of the following subsections defines a distinct model architecture. The user can select and run one of them.

<a id='LSTM1'></a>

### 4.2.1. Shallow LSTM architecture

This a shallow RNN with just one LSTM layer. The same architecture was used for the example by Raschka and Marjili.


```python
# Create a new model
model = tf.keras.Sequential()

# Add an Embedding layer expecting input of the size of the vocabulary, and
# the embedding output dimension
model.add(layers.Embedding(input_dim=vocab_size, output_dim=256)) 

# Add a LSTM layer with 128 internal units
model.add(layers.LSTM(128))

# Add a Dropout layer to avoid overfitting
model.add(layers.Dropout(0.5))

# Add Dense layer as output layer with 1 unit and sigmoid activation
model.add(layers.Dense(1, activation='sigmoid'))
```

<a id='GRU1'></a>

### 4.2.2. Shallow GRU architecture

This is essentially the same shallow RNN as above with a GRU layer instead of the LSTM.


```python
# Create a new model
model = tf.keras.Sequential()

# Add an Embedding layer expecting input of the size of the vocabulary, and
# the embedding output dimension
model.add(layers.Embedding(input_dim=vocab_size, output_dim=256)) 

# Add a GRU layer with 128 internal units
model.add(layers.GRU(128))

# Add a Dropout layer to avoid overfitting
model.add(layers.Dropout(0.5))

# Add Dense layer as output layer with 1 unit and sigmoid activation
model.add(layers.Dense(1, activation='sigmoid'))
```

<a id='LSTM3'></a>

### 4.2.3. Deep LSTM architecture

We can easily deepen our network by stacking a second LSTM layer on top of the first.


```python
# Create a new model
model = tf.keras.Sequential()

# Add an Embedding layer expecting input of the size of the vocabulary, and
# the embedding output dimension
model.add(layers.Embedding(input_dim=vocab_size, output_dim=256)) 

# Add a LSTM layer with 128 internal units
# Return sequences so we can stack the the next LSTM layer on top
model.add(layers.LSTM(128, return_sequences=True))

# Add a second LSTM layer
model.add(layers.LSTM(128))

# Add a Dropout layer to avoid overfitting
model.add(layers.Dropout(0.5))

# Add Dense layer as output layer with 1 unit and sigmoid activation
model.add(layers.Dense(1, activation='sigmoid'))
```

<a id='trainmodel'></a>

## 4.3. Train the model


```python
# print the summary of the model we have defined
print(model.summary())
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, None, 256)         3840000   
    _________________________________________________________________
    lstm_1 (LSTM)                (None, None, 128)         197120    
    _________________________________________________________________
    lstm_2 (LSTM)                (None, 128)               131584    
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 129       
    =================================================================
    Total params: 4,168,833
    Trainable params: 4,168,833
    Non-trainable params: 0
    _________________________________________________________________
    None
    


```python
# compile the model
# we select here the optimizer, loss and metric to be used in training
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
```


```python
# define callbacks for early stopping during training
# stop training when the validation loss `val_accuracy` is no longer improving
callbacks = [tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    min_delta=1e-3,
    patience=5,
    verbose=1,
    restore_best_weights=True)]
```

**WARNING**: as mentioned in the tutorial, the following training routine is computationally intensive. We recommend to sub-sample data in [Section 4.1.](#trainsplit) and/or use HPC infrastructure. 
Note that we ran all the machine learning routines presented in this section on the ETH High Performance Computing (HPC) infrastructure [Euler](https://scicomp.ethz.ch/wiki/Euler), by submitting all jobs to a virtual machine consisting of 32 cores with 3072 MB RAM per core (total RAM: 98.304 GB). Therefore, notebook outputs are not available for the subesquent cells.


```python
# train the model
history = model.fit(X_train, 
                    y_train, 
                    validation_split=0.2,
                    epochs=30, 
                    batch_size=256,
                    callbacks=callbacks,
                    verbose=1)
```


```python
# plot the development of the accuracy over epochs to see the training progress
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
```


```python
# plot the development of the loss over epochs to see the training progress
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
```

<a id='evalmodel'></a>

## 4.4. Evaluate the model on test data


```python
from sklearn.metrics import roc_auc_score

# evalute model to get accuracy and loss on test data
results = model.evaluate(X_test, y_test, batch_size=256, verbose=0)

# calculate AUC on test data
y_pred = model.predict_proba(X_test, batch_size=256)
auc_res = roc_auc_score(y_test, y_pred[:, 0])

print('Test loss:', results[0])
print('Test accuracy:', results[1])
print('Test AUC:', auc_res)
```

<a id='fm'></a>

# 5. Final remarks

The above example RNNs are simple architectures where none of the parameters were optimized for performance. In order to further improve the model accuracy, we could for example
* play around with the network architecture    
     (e.g. the depth of the network, the type of layers used, the number of hidden units within a layer, the activation functions used, ...)
* fine-tune the training parameters   
     (i.e. the number of epochs, batch size, ...)
* perform more elaborate preprocessing on the data  
    (e.g. excluding stopwords, see also the two other Notebooks and Section 1 of the tutorial)
* use a the weights of an already trained embedding for our embedding layer  
    (either as nontrainable fixed weights or with transfer learning, compare the Notebook ```NLP_IMDb_Case_Study_ML.ipynb``` and Section 3 of the tutorial)  

Finally, note that the size of the dataset is arguably still too small to allow for much improvement over the presented architectures and results.

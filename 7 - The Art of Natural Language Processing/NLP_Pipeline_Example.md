# *The Art of Natural Language Processing: NLP Pipeline*

### **Authors: Andrea Ferrario, Mara Nägelin**

**Date: February 2020** (updated September 2020)

Notebook to test NLP preprocessing pipelines, as described in the tutorial `The Art of Natural Language Processing: Classical, Modern and Contemporary Approaches to Text Document Classification'.

# Table of contents
1. [Getting started with Python and Jupyter Notebook](#started)
2. [Test sentence](#test)
3. [NLP preprocessing pipelines](#pipeline)  
    3.1. [Conversion of text to lowercase](#lower)  
    3.2. [Tokenizers](#tokenizers)  
    3.3. [Stopwords removal](#stopwords)  
    3.4. [Part-of-speech tagging](#POS)  
    3.5. [Stemming and lemmatization](#stemming)  

# 1. Getting started with Python and Jupyter Notebook<a name="started"></a>

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


# 2. Test sentence<a name="test"></a>

We introduce the test sentence to be preprocessed with NLP.


```python
text = "In H.P. Lovecraft's short story 'The Call of Cthulhu', the author states that in S. Latitude 47° 9', W. Longitude 126° 43' the great Cthulhu dreams in the sea-bottom city of R'lyeh."
print(text)
```

    In H.P. Lovecraft's short story 'The Call of Cthulhu', the author states that in S. Latitude 47° 9', W. Longitude 126° 43' the great Cthulhu dreams in the sea-bottom city of R'lyeh.
    

We follow the NLP pipeline:
- conversion of text to lowercase;
- tokenization, i.e. split of all strings of text into tokens;
- part-of-speech (POS) tagging of tokenized text;
- stopwords removal;
- stemming or lemmatization

# 3. NLP Preprocessing Pipeline<a name="pipeline"></a>

## 3.1. Conversion of text to lowercase<a name="lower"></a>

We apply lowercase to the test sentence.


```python
text.lower()
```




    "in h.p. lovecraft's short story 'the call of cthulhu', the author states that in s. latitude 47° 9', w. longitude 126° 43' the great cthulhu dreams in the sea-bottom city of r'lyeh."



## 3.2. Tokenizers<a name="tokenizers"></a>


```python
# 1. whitespace tokenizer
#########################
import re
white_tok = text.split()
print(white_tok)
```

    ['In', 'H.P.', "Lovecraft's", 'short', 'story', "'The", 'Call', 'of', "Cthulhu',", 'the', 'author', 'states', 'that', 'in', 'S.', 'Latitude', '47°', "9',", 'W.', 'Longitude', '126°', "43'", 'the', 'great', 'Cthulhu', 'dreams', 'in', 'the', 'sea-bottom', 'city', 'of', "R'lyeh."]
    


```python
# 2. Natural Language Tool Kit tokenizer
########################################
import nltk
from nltk.tokenize import word_tokenize


nltk.download('punkt')

tokens_NLTK = word_tokenize(text)
print(tokens_NLTK)
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\namara\AppData\Roaming\nltk_data...
    [nltk_data]   Unzipping tokenizers\punkt.zip.
    

    ['In', 'H.P', '.', 'Lovecraft', "'s", 'short', 'story', "'The", 'Call', 'of', 'Cthulhu', "'", ',', 'the', 'author', 'states', 'that', 'in', 'S.', 'Latitude', '47°', '9', "'", ',', 'W.', 'Longitude', '126°', '43', "'", 'the', 'great', 'Cthulhu', 'dreams', 'in', 'the', 'sea-bottom', 'city', 'of', "R'lyeh", '.']
    


```python
# 3. SpaCy tokenizer
####################
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
for token in doc:
    print(token.text)
```

    In
    H.P.
    Lovecraft
    's
    short
    story
    '
    The
    Call
    of
    Cthulhu
    '
    ,
    the
    author
    states
    that
    in
    S.
    Latitude
    47
    °
    9
    '
    ,
    W.
    Longitude
    126
    °
    43
    '
    the
    great
    Cthulhu
    dreams
    in
    the
    sea
    -
    bottom
    city
    of
    R'lyeh
    .
    


```python
# 4. sklearn vectorizer
#######################
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

textt = [text]
X = vectorizer.fit_transform(textt)
print(vectorizer.get_feature_names())
```

    ['126', '43', '47', 'author', 'bottom', 'call', 'city', 'cthulhu', 'dreams', 'great', 'in', 'latitude', 'longitude', 'lovecraft', 'lyeh', 'of', 'sea', 'short', 'states', 'story', 'that', 'the']
    

## 3.3. Stopwords removal<a name="stopwords"></a>

We now remove stopwords using NLTK, SpaCy and sklearn.


```python
# 1. stopwords removal with the Natural Language Tool Kit (NLTK)
################################################################
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# we tokenize the test sentence
tokens = word_tokenize(text)


nltk.download('stopwords')

stop = stopwords.words('english')

filtered_tokens = [word for word in tokens if word not in stop]

print('-----------------------------')
print('NLTK tokenized test sentence:', tokens)
print('-----------------------------')
print('NLTK tokenized test sentence after stowords removal:', filtered_tokens)
```

    -----------------------------
    NLTK tokenized test sentence: ['In', 'H.P', '.', 'Lovecraft', "'s", 'short', 'story', "'The", 'Call', 'of', 'Cthulhu', "'", ',', 'the', 'author', 'states', 'that', 'in', 'S.', 'Latitude', '47°', '9', "'", ',', 'W.', 'Longitude', '126°', '43', "'", 'the', 'great', 'Cthulhu', 'dreams', 'in', 'the', 'sea-bottom', 'city', 'of', "R'lyeh", '.']
    -----------------------------
    NLTK tokenized test sentence after stowords removal: ['In', 'H.P', '.', 'Lovecraft', "'s", 'short', 'story', "'The", 'Call', 'Cthulhu', "'", ',', 'author', 'states', 'S.', 'Latitude', '47°', '9', "'", ',', 'W.', 'Longitude', '126°', '43', "'", 'great', 'Cthulhu', 'dreams', 'sea-bottom', 'city', "R'lyeh", '.']
    

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\namara\AppData\Roaming\nltk_data...
    [nltk_data]   Unzipping corpora\stopwords.zip.
    


```python
# removed stopwords
list(set(tokens) - set(filtered_tokens))
```




    ['that', 'in', 'the', 'of']




```python
# 2. stopwords removal with SpaCy
#################################
import spacy

# tokenization
text_spacy = nlp(text)

token_list = []
for token in text_spacy:
    token_list.append(token.text)

# stopwords
from spacy.lang.en.stop_words import STOP_WORDS

# create list of word tokens after removing stopwords note that .vocab() looks at the lexeme of each token 
filtered_sentence =[] 

for word in token_list:
    lexeme = nlp.vocab[word]   
    if lexeme.is_stop == False:
        filtered_sentence.append(word) 
        
print('-----------------------------')
print('SpaCy tokenized test sentence:', token_list)
print('-----------------------------')
print('SpaCy tokenized test sentence after stowords removal:', filtered_sentence) 
```

    -----------------------------
    SpaCy tokenized test sentence: ['In', 'H.P.', 'Lovecraft', "'s", 'short', 'story', "'", 'The', 'Call', 'of', 'Cthulhu', "'", ',', 'the', 'author', 'states', 'that', 'in', 'S.', 'Latitude', '47', '°', '9', "'", ',', 'W.', 'Longitude', '126', '°', '43', "'", 'the', 'great', 'Cthulhu', 'dreams', 'in', 'the', 'sea', '-', 'bottom', 'city', 'of', "R'lyeh", '.']
    -----------------------------
    SpaCy tokenized test sentence after stowords removal: ['In', 'H.P.', 'Lovecraft', "'s", 'short', 'story', "'", 'The', 'Call', 'Cthulhu', "'", ',', 'author', 'states', 'S.', 'Latitude', '47', '°', '9', "'", ',', 'W.', 'Longitude', '126', '°', '43', "'", 'great', 'Cthulhu', 'dreams', 'sea', '-', 'city', "R'lyeh", '.']
    


```python
# removed stopwords
list(set(token_list) - set(filtered_sentence))
```




    ['bottom', 'that', 'the', 'in', 'of']




```python
# 3. stopwords removal with sklearn and TfidfVectorizer()
########################################################
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [text]

vectorizer = TfidfVectorizer()
vectorizer_stop = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(corpus)
X_stop = vectorizer_stop.fit_transform(corpus)

print('-----------------------------')
print('CountVectorizer() tokenized test sentence:', vectorizer.get_feature_names())
print('-----------------------------')
print('CountVectorizer() tokenized test sentence after stowords removal:', vectorizer_stop.get_feature_names()) 
```

    -----------------------------
    CountVectorizer() tokenized test sentence: ['126', '43', '47', 'author', 'bottom', 'call', 'city', 'cthulhu', 'dreams', 'great', 'in', 'latitude', 'longitude', 'lovecraft', 'lyeh', 'of', 'sea', 'short', 'states', 'story', 'that', 'the']
    -----------------------------
    CountVectorizer() tokenized test sentence after stowords removal: ['126', '43', '47', 'author', 'city', 'cthulhu', 'dreams', 'great', 'latitude', 'longitude', 'lovecraft', 'lyeh', 'sea', 'short', 'states', 'story']
    


```python
# removed stopwords
set(vectorizer.get_feature_names()) - set(vectorizer_stop.get_feature_names())
```




    {'bottom', 'call', 'in', 'of', 'that', 'the'}



## 3.4. Part-of-speech tagging<a name="POS"></a>

We perform Part-Of-Speech (POS) tagging using the NLTK.


```python
# introduction of POS tagger per NLTK token (word_tokenize is the tokenizer we choose)
import nltk
from nltk import pos_tag, word_tokenize

nltk.download('averaged_perceptron_tagger')

tokens_NLTK = word_tokenize(text)
print(pos_tag(word_tokenize(text)))
```

    [nltk_data] Downloading package averaged_perceptron_tagger to
    [nltk_data]     C:\Users\namara\AppData\Roaming\nltk_data...
    [nltk_data]   Unzipping taggers\averaged_perceptron_tagger.zip.
    

    [('In', 'IN'), ('H.P', 'NNP'), ('.', '.'), ('Lovecraft', 'NNP'), ("'s", 'POS'), ('short', 'JJ'), ('story', 'NN'), ("'The", 'POS'), ('Call', 'NNP'), ('of', 'IN'), ('Cthulhu', 'NNP'), ("'", 'POS'), (',', ','), ('the', 'DT'), ('author', 'NN'), ('states', 'VBZ'), ('that', 'IN'), ('in', 'IN'), ('S.', 'NNP'), ('Latitude', 'NNP'), ('47°', 'CD'), ('9', 'CD'), ("'", "''"), (',', ','), ('W.', 'NNP'), ('Longitude', 'NNP'), ('126°', 'CD'), ('43', 'CD'), ("'", "''"), ('the', 'DT'), ('great', 'JJ'), ('Cthulhu', 'NNP'), ('dreams', 'NN'), ('in', 'IN'), ('the', 'DT'), ('sea-bottom', 'JJ'), ('city', 'NN'), ('of', 'IN'), ("R'lyeh", 'NNP'), ('.', '.')]
    

## 3.5. Stemming and lemmatization<a name="stemming"></a>

We perform (Porter) stemming and lemmatization on the test sentence, after tokenization.


```python
# NLTK Porter stemming on tokenized test sentence 
#################################################
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in word_tokenize(text)]

# use stemming on NLTK tokenizer text
stem_tokens = tokenizer_porter(text)
print(stem_tokens)
```

    ['In', 'h.p', '.', 'lovecraft', "'s", 'short', 'stori', "'the", 'call', 'of', 'cthulhu', "'", ',', 'the', 'author', 'state', 'that', 'in', 'S.', 'latitud', '47°', '9', "'", ',', 'W.', 'longitud', '126°', '43', "'", 'the', 'great', 'cthulhu', 'dream', 'in', 'the', 'sea-bottom', 'citi', 'of', "r'lyeh", '.']
    


```python
# NLTK lemmatization (WordNet database) on tokenized test sentence
##################################################################
from nltk.stem import WordNetLemmatizer 

nltk.download('wordnet')

# Wordnet lemmatizer
lemmatizer = WordNetLemmatizer()

# NLTK tokenizer
word_list = nltk.word_tokenize(text)

# lemmatization of the list of words and join - we lemmatize verbs (therefore 'v') and we use '***' as separator
lemmatized_output = '***'.join([lemmatizer.lemmatize(w, 'v') for w in word_list])
print(lemmatized_output)
```

    [nltk_data] Downloading package wordnet to
    [nltk_data]     C:\Users\namara\AppData\Roaming\nltk_data...
    [nltk_data]   Unzipping corpora\wordnet.zip.
    

    In***H.P***.***Lovecraft***'s***short***story***'The***Call***of***Cthulhu***'***,***the***author***state***that***in***S.***Latitude***47°***9***'***,***W.***Longitude***126°***43***'***the***great***Cthulhu***dream***in***the***sea-bottom***city***of***R'lyeh***.
    


# *The Art of Natural Language Processing: NLP Pipeline*

### **Authors: Andrea Ferrario, Mara Nägelin**

**Date: February 2020**

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

# 2. Test sentence<a name="test"></a>

We introduce the test sentence to be preprocessed with NLP.


```python
text = "In H.P. Lovecraft's short story 'The Call of Cthulhu', the author states that in S. Latitude 47° 9', W. Longitude 126° 43' the great Cthulhu dreams in the sea-bottom city of R'lyeh."
print(text)
```

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

## 3.2. Tokenizers<a name="tokenizers"></a>


```python
# 1. whitespace tokenizer
#########################
import re
white_tok = text.split()
print(white_tok)
```


```python
# 2. Natural Language Tool Kit tokenizer
########################################
import nltk
from nltk.tokenize import word_tokenize

tokens_NLTK = word_tokenize(text)
print(tokens_NLTK)
```


```python
# 3. SpaCy tokenizer
####################
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
for token in doc:
    print(token.text)
```


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


```python
# removed stopwords
list(set(tokens) - set(filtered_tokens))
```


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


```python
# removed stopwords
list(set(token_list) - set(filtered_sentence))
```


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


```python
# removed stopwords
set(vectorizer.get_feature_names()) - set(vectorizer_stop.get_feature_names())
```

## 3.4. Part-of-speech tagging<a name="POS"></a>

We perform Part-Of-Speech (POS) tagging using the NLTK.


```python
# introduction of POS tagger per NLTK token (word_tokenize is the tokenizer we choose)
import nltk
from nltk import pos_tag, word_tokenize

tokens_NLTK = word_tokenize(text)
print(pos_tag(word_tokenize(text)))
```

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


```python
# NLTK lemmatization (WordNet database) on tokenized test sentence
##################################################################
from nltk.stem import WordNetLemmatizer 

# Wordnet lemmatizer
lemmatizer = WordNetLemmatizer()

# NLTK tokenizer
word_list = nltk.word_tokenize(text)

# lemmatization of the list of words and join - we lemmatize verbs (therefore 'v') and we use '***' as separator
lemmatized_output = '***'.join([lemmatizer.lemmatize(w, 'v') for w in word_list])
print(lemmatized_output)
```

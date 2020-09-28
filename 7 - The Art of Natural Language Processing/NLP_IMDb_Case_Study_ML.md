# *The Art of Natural Language Processing: Machine Learning for the Case Study*

### **Authors: Andrea Ferrario, Mara Nägelin**

**Date: February 2020** (updated September 2020)

Notebook to run the machine learning modeling in the Classical and Modern Approaches, as described in the tutorial `The Art of Natural Language Processing: Classical, Modern and Contemporary Approaches to Text Document Classification'.

# Table of contents
1. [Getting started with Python and Jupyter Notebook](#started)
2. [Import data](#import)
3. [Duplicated reviews](#duplicated)
4. [Data preprocessing](#preprocessing)
5. [POS-tagging](#POS)
6. [Pre-trained word embeddings](#emb)
7. [Data analytics](#analytics)  
    7.1. [A quick check of data structure](#check)  
    7.2. [Basic linguistic analysis of movie reviews](#basic)
8. [Machine learning](#ML)  
    8.1. [Adaptive boosting (ADA)](#ADA)  
    .......8.1.1. [Bag-of-words](#ADA_BOW)  
    .......8.1.2. [Bag-of-POS](#ADA_BOP)  
    .......8.1.3. [Embeddings](#ADA_E)  
    8.2. [Random forests (RF)](#RF)  
    .......8.2.1. [Bag-of-words](#RF_BOW)  
    .......8.2.2. [Bag-of-POS](#RF_BOP)  
    .......8.2.3. [Embeddings](#RF_E)  
    8.3. [Extreme gradient boosting (XGB)](#XGB)  
    .......8.3.1. [Bag-of-words](#XGB_BOW)  
    .......8.3.2. [Bag-of-POS](#XGB_BOP)  
    .......8.3.3. [Embeddings](#XGB_E)

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


# 2. Import data<a name="import"></a>


```python
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
```

    0% [##############################] 100% | ETA: 00:00:00
    Total time elapsed: 00:01:54
    

# 3. Duplicated reviews<a name="duplicated"></a>


```python
# check for duplicates
duplicates = df[df.duplicated()]  #equivalent to keep = first. Duplicated rows, except the first entry, are marked as 'True'
print(len(duplicates))
```

    418
    


```python
# a check on the duplicated review
duplicates.review   
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
# remove duplicates: 
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



# 4. Data preprocessing<a name="preprocessing"></a>


```python
# an example of 'raw' review: we have all sort of HTML markup
df.loc[500, 'review']
```




    'I have always liked this film and I\'m glad it\'s available finally on DVD so more viewers can see what I have been telling them all these years. Story is about a high school virgin named Gary (Lawrence Monoson) who works at a pizza place as a delivery boy and he hangs out with his friends David (Joe Rubbo) and Rick (Steve Antin). Gary notices Karen (Diane Franklin) who is the new girl in school and one morning he gives her a ride and by this time he is totally in love. That night at a party he see\'s Rick with Karen and now he is jealous of his best friend but doesn\'t tell anyone of his true feelings.<br /><br />*****SPOILER ALERT*****<br /><br />Rick asks Gary if he can borrow his Grandmothers vacant home but Gary makes up an excuse so that Rick can\'t get Karen alone. But one night Rick brags to Gary that he nailed her at the football field and Gary becomes enraged. A few days later in the school library Gary see\'s Rick and Karen arguing and he asks Karen what is wrong. She tells him that she\'s pregnant and that Rick has dumped her. Gary helps her by taking her to his Grandmothers home and paying for her abortion. Finally, Gary tells Karen how he really feels about her and she seems receptive to his feelings but later at her birthday party he walks in on Karen and Rick together again. Gary drives off without the girl! This film ends with a much more realistic version of how life really is. No matter how nice you are you don\'t necessarily get the girl.<br /><br />This film was directed by Boaz Davidson who would go on to be a pretty competent action film director and he did two things right with this movie. First, he made sure that there was plenty of gratuitous nudity so that this was marketable to the young males that usually go to these films. Secondly, he had the film end with young Gary without Karen and I think the males in the audience can relate to being screwed over no matter how hard you try and win a girls heart. Yes, this film is silly and exploitive but it is funny and sexy. Actress Louisa Moritz almost steals the film as the sexy Carmela. Moritz was always a popular "B" level actress and you might remember her in "One Flew Over The Cuckoo\'s Nest". Like "Fast Times at Ridgemont High" this has a very good soundtrack and the songs being played reflect what is going on in the story. But at the heart of this film is two very good performances by Monoson and Franklin. There is nudity required by Franklin but she still conveys the sorrow of a young girl who gets dumped at a crucial time. She\'s always been a good actress and her natural charm is very evident in this film. But this is still Monoson\'s story and you can\'t help but feel for this guy. When the film ends it\'s his performance that stays with you. It\'s a solid job of acting that makes this more than just a teen sex comedy. Even with the silly scenarios of teens trying to have sex this film still manages to achieve what it wants. Underrated comedy hits the bullseye.'




```python
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
```


```python
# checking again the same review
df.loc[500, 'review']
```




    'i have always liked this film and i m glad it s available finally on dvd so more viewers can see what i have been telling them all these years story is about a high school virgin named gary lawrence monoson who works at a pizza place as a delivery boy and he hangs out with his friends david joe rubbo and rick steve antin gary notices karen diane franklin who is the new girl in school and one morning he gives her a ride and by this time he is totally in love that night at a party he see s rick with karen and now he is jealous of his best friend but doesn t tell anyone of his true feelings spoiler alert rick asks gary if he can borrow his grandmothers vacant home but gary makes up an excuse so that rick can t get karen alone but one night rick brags to gary that he nailed her at the football field and gary becomes enraged a few days later in the school library gary see s rick and karen arguing and he asks karen what is wrong she tells him that she s pregnant and that rick has dumped her gary helps her by taking her to his grandmothers home and paying for her abortion finally gary tells karen how he really feels about her and she seems receptive to his feelings but later at her birthday party he walks in on karen and rick together again gary drives off without the girl this film ends with a much more realistic version of how life really is no matter how nice you are you don t necessarily get the girl this film was directed by boaz davidson who would go on to be a pretty competent action film director and he did two things right with this movie first he made sure that there was plenty of gratuitous nudity so that this was marketable to the young males that usually go to these films secondly he had the film end with young gary without karen and i think the males in the audience can relate to being screwed over no matter how hard you try and win a girls heart yes this film is silly and exploitive but it is funny and sexy actress louisa moritz almost steals the film as the sexy carmela moritz was always a popular b level actress and you might remember her in one flew over the cuckoo s nest like fast times at ridgemont high this has a very good soundtrack and the songs being played reflect what is going on in the story but at the heart of this film is two very good performances by monoson and franklin there is nudity required by franklin but she still conveys the sorrow of a young girl who gets dumped at a crucial time she s always been a good actress and her natural charm is very evident in this film but this is still monoson s story and you can t help but feel for this guy when the film ends it s his performance that stays with you it s a solid job of acting that makes this more than just a teen sex comedy even with the silly scenarios of teens trying to have sex this film still manages to achieve what it wants underrated comedy hits the bullseye '




```python
# save preprocessed data as csv 
path = '...'  # insert path
df.to_csv(path, index=False, encoding='utf-8')
```

# 5. POS - tagging<a name="POS"></a>


```python
# we apply POS-tagging on (deduplicated and) pre-processed data - let us import them
path = '...' # insert path
df = pd.read_csv(path, encoding='utf-8')
df.shape
```




    (49582, 2)




```python
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
```


```python
# save POS-tagged data as csv 
path = '...' # insert path 
df.to_csv(path, index=False, encoding='utf-8')
```

# 6. Pre-trained word embeddings<a name="emb"></a>


```python
# we apply embeddings on de-duplicated and pre-processed data - let us import them
path = '...' # insert path
df = pd.read_csv(path, encoding='utf-8')
df.shape
```




    (49582, 2)




```python
# import pre-trained word embedding model
import spacy
nlp = spacy.load('en_core_web_md') # load the model first if necessary: python -m spacy download en_core_web_md
```


```python
# we stack (like a numpy vertical stack) the 300 variables obtained from averaging the embedding of each df.review entry
# WARNING: this is computationally expensive. Alternatively try with the smaller model en_core_web_sm
import numpy as np
emb = np.vstack(df.review.apply(lambda x: nlp(x).vector))
```


```python
# embeddings into a dataframe
emb = pd.DataFrame(emb, columns = np.array([str(x) for x in range(0, 299 + 1)]) )
print(emb.shape)
print(emb.columns.values)
```

    (49582, 300)
    ['0' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15'
     '16' '17' '18' '19' '20' '21' '22' '23' '24' '25' '26' '27' '28' '29'
     '30' '31' '32' '33' '34' '35' '36' '37' '38' '39' '40' '41' '42' '43'
     '44' '45' '46' '47' '48' '49' '50' '51' '52' '53' '54' '55' '56' '57'
     '58' '59' '60' '61' '62' '63' '64' '65' '66' '67' '68' '69' '70' '71'
     '72' '73' '74' '75' '76' '77' '78' '79' '80' '81' '82' '83' '84' '85'
     '86' '87' '88' '89' '90' '91' '92' '93' '94' '95' '96' '97' '98' '99'
     '100' '101' '102' '103' '104' '105' '106' '107' '108' '109' '110' '111'
     '112' '113' '114' '115' '116' '117' '118' '119' '120' '121' '122' '123'
     '124' '125' '126' '127' '128' '129' '130' '131' '132' '133' '134' '135'
     '136' '137' '138' '139' '140' '141' '142' '143' '144' '145' '146' '147'
     '148' '149' '150' '151' '152' '153' '154' '155' '156' '157' '158' '159'
     '160' '161' '162' '163' '164' '165' '166' '167' '168' '169' '170' '171'
     '172' '173' '174' '175' '176' '177' '178' '179' '180' '181' '182' '183'
     '184' '185' '186' '187' '188' '189' '190' '191' '192' '193' '194' '195'
     '196' '197' '198' '199' '200' '201' '202' '203' '204' '205' '206' '207'
     '208' '209' '210' '211' '212' '213' '214' '215' '216' '217' '218' '219'
     '220' '221' '222' '223' '224' '225' '226' '227' '228' '229' '230' '231'
     '232' '233' '234' '235' '236' '237' '238' '239' '240' '241' '242' '243'
     '244' '245' '246' '247' '248' '249' '250' '251' '252' '253' '254' '255'
     '256' '257' '258' '259' '260' '261' '262' '263' '264' '265' '266' '267'
     '268' '269' '270' '271' '272' '273' '274' '275' '276' '277' '278' '279'
     '280' '281' '282' '283' '284' '285' '286' '287' '288' '289' '290' '291'
     '292' '293' '294' '295' '296' '297' '298' '299']
    


```python
# join embeddings with dataframe
df_embed = pd.concat([df, emb], axis=1)
```


```python
# check the shape of the resulting dataframe
df_embed.shape
```




    (49582, 302)




```python
# save word embedding data as csv 
path = '...' # insert path
df_embed.to_csv(path, index=False, encoding='utf-8')
```

# 7. Data analytics<a name="analytics"></a>

We reproduce main data analytics results in Section 6.3 of the tutorial. We use the preprocessed and deduplicated data, for simplicity.

## 7.1. A quick check of data structure<a name="check"></a>


```python
# importing data
import pandas as pd

path = '...' # insert path for deduplicated and preprocessed data
df = pd.read_csv(path)
```


```python
# imported data structure
df.shape
```




    (49582, 2)




```python
# columns in data
df.columns
```




    Index(['review', 'sentiment'], dtype='object')




```python
# imported data: first 10 entries
df.head(10)
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
    <tr>
      <td>0</td>
      <td>i went and saw this movie last night after bei...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>actor turned director bill paxton follows up h...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>as a recreational golfer with some knowledge o...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>i saw this film in a sneak preview and it is d...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>bill paxton has taken the true story of the 19...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>5</td>
      <td>i saw this film on september 1st 2005 in india...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>6</td>
      <td>maybe i m reading into this too much but i won...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>7</td>
      <td>i felt this film did have many good qualities ...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>8</td>
      <td>this movie is amazing because the fact that th...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>9</td>
      <td>quitting may be as much about exiting a pre o...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# counts of rviews per sentiment value
df.sentiment.value_counts()
```




    1    24884
    0    24698
    Name: sentiment, dtype: int64



## 7.2. Basic linguistic analysis of movie reviews<a name="basic"></a>


```python
# show distribution of review lenghts 
# we strip leading and trailing whitespaces and tokenize by whitespace
df['word_count'] = df['review'].apply(lambda x: len(x.strip().split(" ")))
df[['review','sentiment', 'word_count']].head(10)
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
      <th>word_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>i went and saw this movie last night after bei...</td>
      <td>1</td>
      <td>153</td>
    </tr>
    <tr>
      <td>1</td>
      <td>actor turned director bill paxton follows up h...</td>
      <td>1</td>
      <td>353</td>
    </tr>
    <tr>
      <td>2</td>
      <td>as a recreational golfer with some knowledge o...</td>
      <td>1</td>
      <td>247</td>
    </tr>
    <tr>
      <td>3</td>
      <td>i saw this film in a sneak preview and it is d...</td>
      <td>1</td>
      <td>128</td>
    </tr>
    <tr>
      <td>4</td>
      <td>bill paxton has taken the true story of the 19...</td>
      <td>1</td>
      <td>206</td>
    </tr>
    <tr>
      <td>5</td>
      <td>i saw this film on september 1st 2005 in india...</td>
      <td>1</td>
      <td>318</td>
    </tr>
    <tr>
      <td>6</td>
      <td>maybe i m reading into this too much but i won...</td>
      <td>1</td>
      <td>344</td>
    </tr>
    <tr>
      <td>7</td>
      <td>i felt this film did have many good qualities ...</td>
      <td>1</td>
      <td>144</td>
    </tr>
    <tr>
      <td>8</td>
      <td>this movie is amazing because the fact that th...</td>
      <td>1</td>
      <td>174</td>
    </tr>
    <tr>
      <td>9</td>
      <td>quitting may be as much about exiting a pre o...</td>
      <td>1</td>
      <td>959</td>
    </tr>
  </tbody>
</table>
</div>




```python
# summary statistics of word counts
print(df['word_count'].describe())
```

    count    49582.000000
    mean       235.660340
    std        174.444773
    min          6.000000
    25%        129.000000
    50%        177.000000
    75%        286.000000
    max       2498.000000
    Name: word_count, dtype: float64
    


```python
# show histograms of word counts divided per sentiment
from matplotlib import pyplot

x = df[df['sentiment']==0].word_count
y = df[df['sentiment']==1].word_count

pyplot.hist(x, bins=50, alpha=0.5, label='negative sentiment reviews')
pyplot.hist(y, bins=50, alpha=0.5, label='positive sentiment reviews')
pyplot.legend(loc='upper right')
pyplot.show()
```


    
![png](output_43_0.png)
    



```python
# summary of distributions of word counts
print(x.describe())
print(y.describe())
```

    count    24698.000000
    mean       234.109847
    std        168.079121
    min          6.000000
    25%        131.000000
    50%        178.000000
    75%        283.000000
    max       1550.000000
    Name: word_count, dtype: float64
    count    24884.000000
    mean       237.199244
    std        180.531262
    min         10.000000
    25%        127.000000
    50%        176.000000
    75%        288.000000
    max       2498.000000
    Name: word_count, dtype: float64
    


```python
# some checks (e.g. word_counts=6 or 1550 or 2498 )
df[df['word_count']==6]
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
      <th>word_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>19476</td>
      <td>primary plot primary direction poor interpreta...</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <td>21333</td>
      <td>read the book forget the movie</td>
      <td>0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# average word length (again, we tokenize by whitespaces)
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

df['avg_word'] = df['review'].apply(lambda x: avg_word(x.strip()))
df[['review','word_count', 'sentiment', 'avg_word']].head(10)
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
      <th>word_count</th>
      <th>sentiment</th>
      <th>avg_word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>i went and saw this movie last night after bei...</td>
      <td>153</td>
      <td>1</td>
      <td>4.091503</td>
    </tr>
    <tr>
      <td>1</td>
      <td>actor turned director bill paxton follows up h...</td>
      <td>353</td>
      <td>1</td>
      <td>4.501416</td>
    </tr>
    <tr>
      <td>2</td>
      <td>as a recreational golfer with some knowledge o...</td>
      <td>247</td>
      <td>1</td>
      <td>4.607287</td>
    </tr>
    <tr>
      <td>3</td>
      <td>i saw this film in a sneak preview and it is d...</td>
      <td>128</td>
      <td>1</td>
      <td>4.085938</td>
    </tr>
    <tr>
      <td>4</td>
      <td>bill paxton has taken the true story of the 19...</td>
      <td>206</td>
      <td>1</td>
      <td>4.723301</td>
    </tr>
    <tr>
      <td>5</td>
      <td>i saw this film on september 1st 2005 in india...</td>
      <td>318</td>
      <td>1</td>
      <td>4.544025</td>
    </tr>
    <tr>
      <td>6</td>
      <td>maybe i m reading into this too much but i won...</td>
      <td>344</td>
      <td>1</td>
      <td>4.270349</td>
    </tr>
    <tr>
      <td>7</td>
      <td>i felt this film did have many good qualities ...</td>
      <td>144</td>
      <td>1</td>
      <td>4.652778</td>
    </tr>
    <tr>
      <td>8</td>
      <td>this movie is amazing because the fact that th...</td>
      <td>174</td>
      <td>1</td>
      <td>4.436782</td>
    </tr>
    <tr>
      <td>9</td>
      <td>quitting may be as much about exiting a pre o...</td>
      <td>959</td>
      <td>1</td>
      <td>4.503650</td>
    </tr>
  </tbody>
</table>
</div>




```python
# distributions of word lengths conditional per sentiment
x = df[df['sentiment']==0].avg_word
y = df[df['sentiment']==1].avg_word
print(x.describe())
print()
print(y.describe())
```

    count    24698.000000
    mean         4.266094
    std          0.287540
    min          2.917808
    25%          4.073807
    50%          4.253968
    75%          4.447635
    max          7.500000
    Name: avg_word, dtype: float64
    
    count    24884.000000
    mean         4.325630
    std          0.318722
    min          3.137931
    25%          4.114986
    50%          4.316667
    75%          4.526851
    max         11.673077
    Name: avg_word, dtype: float64
    


```python
# some checks (e.g. avg_word>=11)
df[df['avg_word']>=11]
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
      <th>word_count</th>
      <th>avg_word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>11985</td>
      <td>whoops looks like it s gonna cost you a whopp...</td>
      <td>1</td>
      <td>52</td>
      <td>11.673077</td>
    </tr>
  </tbody>
</table>
</div>




```python
# stop words statistics - stopword from NLTK
from nltk.corpus import stopwords
stop = stopwords.words('english')

df['stopwords'] = df['review'].apply(lambda x: len([x for x in x.strip().split() if x in stop]))
df[['review','word_count', 'sentiment', 'avg_word', 'stopwords']].head(10)
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
      <th>word_count</th>
      <th>sentiment</th>
      <th>avg_word</th>
      <th>stopwords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>i went and saw this movie last night after bei...</td>
      <td>153</td>
      <td>1</td>
      <td>4.091503</td>
      <td>82</td>
    </tr>
    <tr>
      <td>1</td>
      <td>actor turned director bill paxton follows up h...</td>
      <td>353</td>
      <td>1</td>
      <td>4.501416</td>
      <td>167</td>
    </tr>
    <tr>
      <td>2</td>
      <td>as a recreational golfer with some knowledge o...</td>
      <td>247</td>
      <td>1</td>
      <td>4.607287</td>
      <td>123</td>
    </tr>
    <tr>
      <td>3</td>
      <td>i saw this film in a sneak preview and it is d...</td>
      <td>128</td>
      <td>1</td>
      <td>4.085938</td>
      <td>71</td>
    </tr>
    <tr>
      <td>4</td>
      <td>bill paxton has taken the true story of the 19...</td>
      <td>206</td>
      <td>1</td>
      <td>4.723301</td>
      <td>91</td>
    </tr>
    <tr>
      <td>5</td>
      <td>i saw this film on september 1st 2005 in india...</td>
      <td>318</td>
      <td>1</td>
      <td>4.544025</td>
      <td>144</td>
    </tr>
    <tr>
      <td>6</td>
      <td>maybe i m reading into this too much but i won...</td>
      <td>344</td>
      <td>1</td>
      <td>4.270349</td>
      <td>184</td>
    </tr>
    <tr>
      <td>7</td>
      <td>i felt this film did have many good qualities ...</td>
      <td>144</td>
      <td>1</td>
      <td>4.652778</td>
      <td>68</td>
    </tr>
    <tr>
      <td>8</td>
      <td>this movie is amazing because the fact that th...</td>
      <td>174</td>
      <td>1</td>
      <td>4.436782</td>
      <td>93</td>
    </tr>
    <tr>
      <td>9</td>
      <td>quitting may be as much about exiting a pre o...</td>
      <td>959</td>
      <td>1</td>
      <td>4.503650</td>
      <td>460</td>
    </tr>
  </tbody>
</table>
</div>




```python
# distributions of stop words conditional per sentiment
x = df[df['sentiment']==0].stopwords
y = df[df['sentiment']==1].stopwords
print(x.describe())
print()
print(y.describe())
```

    count    24698.000000
    mean       116.211556
    std         83.499135
    min          0.000000
    25%         65.000000
    50%         89.000000
    75%        140.000000
    max        726.000000
    Name: stopwords, dtype: float64
    
    count    24884.000000
    mean       115.732238
    std         87.736974
    min          3.000000
    25%         62.000000
    50%         87.000000
    75%        140.000000
    max       1208.000000
    Name: stopwords, dtype: float64
    


```python
# some checks (e.g. stopwords==0)
df[df['stopwords']==0]
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
      <th>word_count</th>
      <th>avg_word</th>
      <th>stopwords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>19476</td>
      <td>primary plot primary direction poor interpreta...</td>
      <td>0</td>
      <td>6</td>
      <td>7.5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# 8. Machine Learning<a name="ML"></a>

We replicate the machine learning pipelines from the tutorial, Section 6.4 (Classical and Modern Approaches).

**WARNING**: as mentioned in the tutorial, the following cross-validation routines are computationally intensive. We recommend to sub-sample data and/or use HPC infrastructure (specifying the parameter njobs in GridSearch() accordingly). Test runs can be launched on reduced hyperparameter grids, as well. 
Note that we ran all the machine learning routines presented in this section on the ETH High Performance Computing (HPC) infrastructure [Euler](https://scicomp.ethz.ch/wiki/Euler), by submitting all jobs to a virtual machine consisting of 32 cores with 3072 MB RAM per core (total RAM: 98.304 GB). Therefore, notebook cell outputs are not available for this section.

## 8.1. Adaptive boosting (ADA)<a name="ADA"></a>

We use the adaptive boosting (ADA) algorithm on top of NLP pipelines (bag-of models and pre-trained word embeddings).

### 8.1.1. Bag-of-words<a name="ADA_BOW"></a>


```python
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

path = '...'  # insert path to deduplicated and preprocessed data
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
```

### 8.1.2. Bag-of-POS<a name="ADA_BOP"></a>


```python
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
```

### 8.1.3. Embeddings<a name="ADA_E"></a>


```python
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
```

## 8.2. Random Forests (RF)<a name="RF"></a>

We use the random forests (RF) algorithm on top of NLP pipelines (bag-of models and pre-trained word embeddings).

### 8.2.1. Bag-of-words<a name="RF_BOW"></a>


```python
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

path = '...'  # insert path to deduplicated and preprocessed data
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
```

### 8.2.2. Bag-of-POS<a name="RF_BOP"></a>


```python
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
```

### 8.2.3. Embeddings<a name="RF_E"></a>


```python
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
```

## 8.3. Extreme gradient boosting (XGB)<a name="XGB"></a>

We use the extreme gradient boosting (XGB) algorithm on top of NLP pipelines (bag-of models and pre-trained word embeddings). We can use the cell below to install xgboost, if other imports failed.


```python
# importing xgboost REMARK: run this cell only if other imports failed. Delete it in case xgboost has been already imported
import pip
pip.main(['install', 'xgboost'])
```

### 8.3.1. Bag-of-words<a name="XGB_BOW"></a>


```python
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
```

### 8.3.2. Bag-of-POS<a name="XGB_BOP"></a>


```python
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
```

### 8.3.3. Embeddings<a name="XGB_E"></a>


```python
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
```

import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import pickle
import os
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression


class Document:
    def __init__(self, title, text):
        self.title = title
        self.text = text

    def format(self, query):
        return [self.title, self.text[:200] + ' ...']


def load_data(path):
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return [], {}, []


def load_model(path):
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None, None


documents, index, text = load_data('index.pickle')
model, tfv = load_model('model.pickle')
nltk.download('stopwords')
sw_eng = stopwords.words('english')
stemmer = SnowballStemmer(language='english')


def build_index():
    if len(documents) == 0:
        data = pd.read_csv('lyrics-data.csv')
        artist_data = pd.read_csv('artists-data.csv')
        data = pd.merge(data, artist_data, left_on='ALink', right_on='Link', how='inner')[
            ['SName', 'Lyric', 'Artist']].drop_duplicates(keep='first')

        for i, row in data.iterrows():
            documents.append(Document((str(row['SName'])) + ' - ' + str(row['Artist']),
                                      str(row['Lyric'])))

        for idx, doc in enumerate(documents):
            text.append('')
            for word in set(re.split(r'[^a-z0-9]', (doc.title + ' ' + doc.text).lower())):
                if word not in sw_eng:
                    stemmed_word = stemmer.stem(word)
                    text[idx] += stemmed_word + ' '
                    if stemmed_word not in index:
                        index[stemmed_word] = []
                    index[stemmed_word].append(idx)
        with open('index.pickle', "wb") as f:
            pickle.dump((documents, index, text), f)


def score(query, document):
    stemmed_doc = ' '.join(
        [stemmer.stem(word) for word in set(re.split(r'[^a-z0-9]', (document.title + ' ' + document.text).lower())) if
         word not in sw_eng])
    keywords = [stemmer.stem(word) for word in re.split(r'[^a-z0-9]', query.lower()) if word not in sw_eng]
    intersection = ' '.join(set(stemmed_doc.split()).intersection(keywords))
    if len(intersection) == 0:
        return 0
    data = 2*tfv.transform([intersection]).toarray()+tfv.transform([stemmed_doc]).toarray()
    return 1 / (1 + np.exp(-model.predict(data)))


def retrieve(query):
    keywords = [stemmer.stem(word) for word in re.split(r'[^a-z0-9]', query.lower()) if word not in sw_eng]
    keywords = list(set(index.keys()).intersection(keywords))
    if len(keywords) == 0:
        return documents[:20], [range(20)]
    s = set(index[keywords[0]])
    for word in keywords[1:]:
        s = s.intersection(index[word])

    candidates = [documents[i] for i in s]
    return candidates[:20], list(s)[:20]

#!/usr/bin/python
# -*- coding: utf-8 -*-
# The purpose of this file is to import the Hashing Vectorizer into to python session when needed.

from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

# Set current working directory

cur_dir = os.path.dirname(__file__)

# Load stop words from the pickle to be used in the tokenizer.

stop = pickle.load(open(os.path.join(cur_dir, 'pkl_objects',
                   'stopwords.pkl'), 'rb'))


# As defined in the notebook.

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) \
        + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


# Define Hashing Vectorizer object.

vect = HashingVectorizer(decode_error='ignore', n_features=2 ** 21,
                         preprocessor=None, tokenizer=tokenizer)  # we set a large value for number of features to reduce hash collisions


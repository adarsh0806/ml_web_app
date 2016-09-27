# Mini version of the web app. Creates the pickles using the test dataset which are then loaded in pickle_load.py 
# for making predictions.
import pickle
import os
import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

stop = stopwords.words('english')

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# Define Hashing Vectorizer object
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)

clf = SGDClassifier(loss='log', random_state=1, n_iter=1)

# Load sample dataset
df = pd.read_csv('./sample_movie_review.csv', encoding='utf-8')

# Split data
X_train = df['review'].values
y_train = df['sentiment'].values

# Train the classifier
X_train = vect.transform(X_train)
clf.fit(X_train, y_train)

# Create the stopwords pickle
pickle.dump(stop,
            open('stopwords.pkl', 'wb'),
            protocol=2)

# Create the classifier pickle
pickle.dump(clf,
            open('classifier.pkl', 'wb'),
            protocol=2)
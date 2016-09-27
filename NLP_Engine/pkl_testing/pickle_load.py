import pickle
import os
import re
from vectorizer import vect
import numpy as np

# load the pickled classifier created using the full dataset
# clf = pickle.load(open(os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))

# load the pickled classifier created using the smaller test dataset
clf = pickle.load(open(os.path.join('classifier.pkl'), 'rb'))

# define the labels
label = {0: 'negative', 1: 'positive'}

# test review
test = 'i love this movie'

X = vect.transform(test)

print 'Prediction: ', label[clf.predict(X)[0]]
print '\nProbability: ', np.max(clf.predict_proba(X)) * 100
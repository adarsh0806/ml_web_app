import pickle
import sqlite3
import numpy as np
import os

# import HashingVectorizer from local dir
from vectorizer import vect

def update_model(db_path, model, batch_size=5000):
    # connect to the db
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT * from review_db')

    # collect the last 5000 entries
    results = c.fetchmany(batch_size)
    while results:
        data = np.array(results)
        # X -> review
        X = data[:, 0]
        # y -> positive or negative
        y = data[:, 1].astype(int)

        classes = np.array([0, 1])
        X_train = vect.transform(X)
        model.partial_fit(X_train, y, classes=classes)
        results = c.fetchmany(batch_size)

    conn.close()
    return model

cur_dir = os.path.dirname(__file__)

# load the classifier pickle
clf = pickle.load(open(os.path.join(cur_dir,
                  'pkl_objects',
                  'classifier.pkl'), 'rb'))

db = os.path.join(cur_dir, 'reviews.sqlite')

# update the classifier with the new data
clf = update_model(db_path=db, model=clf, batch_size=5000)

# create new classifier pickle
# pickle.dump(clf, 
#             open(os.path.join(cur_dir,'pkl_objects', 'classifier.pkl'), 'wb')
#             , protocol=2)
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
from update import update_model

# import HashingVectorizer from vectorizer.py
from vectorizer import vect

app = Flask(__name__)

# set the current directory
cur_dir = os.path.dirname(__file__)

# load the pickled classifier
clf = pickle.load(open(os.path.join(cur_dir,
                 'pkl_objects',
                 'classifier.pkl'), 'rb'))

# set up the path for the database
db = os.path.join(cur_dir, 'reviews.sqlite')

# classify function to return the predicted class label as well as the probability of correctness of the prediction
def classify(document):
	label = {0: 'negative', 1: 'positive'}
	# remove stop words and tokenize
	X = vect.transform([document])
	# make the prediction on the review, which is the first item 
	y = clf.predict(X)[0]
	# calculate probability of of correctness
	proba = np.max(clf.predict_proba(X))
	# return the label of y
	return label[y], proba

def train(document, y):
	# remove stop words and tokenize
	X = vect.transform([document])
	clf.partial_fit(X, [y])

def sqlite_entry(path, document, y):
	conn = sqlite3.connect(path)
	c = conn.cursor()
	c.execute("INSERT INTO review_db (review, sentiment, date)"\
		" VALUES (?, ?, DATETIME('now'))", (document, y))
	conn.commit()
	conn.close()

#### Flask set up ####

class ReviewForm(Form):
	# set up the form to take review input
	moviereview = TextAreaField('', [validators.DataRequired(), validators.length(min = 15)])

@app.route('/')
def index():
	form = ReviewForm(request.form)
	# return the page to take input
	return render_template('reviewform.html', 
		form = form)

@app.route('/results', methods = ['POST'])
def results():
	form = ReviewForm(request.form)
	if request.method == 'POST' and form.validate():
		# review is the input given by the user
		review = request.form['moviereview']
		# run classifier on the input given by the user
		y, proba = classify(review)
		# return the results page
		return render_template('results.html',
			content = review,
			prediction = y,
			probability = round(proba * 100, 2))
	return render_template('reviewform.html', 
		form = form)

@app.route('/thanks', methods = ['POST'])
def feedback():
	# on click of the feedback button in results.html
	feedback = request.form['feedback_button']
	review = request.form['review']
	prediction = request.form['prediction']
	# inverse label
	inv_label = {'negative': 0, 'positive': 1}
	# take in the numerical value of the user input
	y = inv_label[prediction]
	if feedback == 'Incorrect':
		# set the prediction to the correct value
		y = int(not(y))
	# train the model with the correct value
	train(review, y)
	# create new db entry with the correct entry
	sqlite_entry(db, review, y)
	return render_template('thanks.html')

if __name__ == '__main__':
	# clf = update_model(db_path= "db", model= clf, batch_size= 5000)
	app.run(debug = True)
	


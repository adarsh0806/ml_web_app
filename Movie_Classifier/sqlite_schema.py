import sqlite3
import os
conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()
# create the table
c.execute('CREATE TABLE review_db'\
	'(review TEXT, sentiment INTEGER, date TEXT)')

example1 = 'i love this movie'

# insert value into the table
c.execute("INSERT INTO review_db"\
	"(review, sentiment, date) VALUES"\
	"(?,?,DATETIME('now'))", (example1, 1))

example2 = 'i hated this movie'

c.execute("INSERT INTO review_db"\
	"(review, sentiment, date) VALUES"\
	"(?,?,DATETIME('now'))", (example2, 0))

conn.commit()
conn.close()
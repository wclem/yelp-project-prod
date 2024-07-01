import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib


# Create the connection
cnx = sqlite3.connect(r'rev_db3.db')
curs = cnx.cursor()


# create the dataframe from a query
df = pd.read_sql_query("SELECT * FROM data", cnx)
curs.execute("SELECT COUNT(*) FROM data").fetchall()



vectorizer = CountVectorizer(min_df=1)
text = df["text"]


custom_stopwords = {'don', "don't", 'ain', "aren't", 'couldn', "couldn't",
                   'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
                   'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
                   'mightn', "mightn't", 'mustn', "musn't", 'needn', "needn't", 
                   'shan', "shan't", 'no', 'nor', 'not', 'shouldn', "shouldn't",
                   'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}

corpus = []
ps = PorterStemmer()
stop_words = set(stopwords.words("english")) - custom_stopwords

for i in range(100):
    review = re.sub('[^a-zA-Z]', ' ',text[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    review = " ".join(review)
    corpus.append(review)

text = text.head(10000)

# call `fit` to build the vocabulary
vectorizer.fit(text)

# call `transform` to convert text to a bag of words
x = vectorizer.transform(text)

# CountVectorizer uses a sparse array to save memory, but it's easier in this assignment to 
# convert back to a "normal" numpy array
x = x.toarray()

y=df["stars"].head(10000)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)



joblib.dump(nb, "yelpmodel.pkl")

joblib.dump(vectorizer, "yelp_vectorizer.pkl")

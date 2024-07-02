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
import time

def load_data():
    start = time.time()
    cnx = sqlite3.connect(r'rev_db3.db')
    curs = cnx.cursor()
    df = pd.read_sql_query("SELECT * FROM data", cnx)
    end = time.time()
    print(f"Loading time: {end - start} sec.")
    return df

def clean_text(input_df):
    custom_stopwords = {'don', "don't", 'ain', "aren't", 'couldn', "couldn't",
                    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
                    'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
                    'mightn', "mightn't", 'mustn', "musn't", 'needn', "needn't", 
                    'shan', "shan't", 'no', 'nor', 'not', 'shouldn', "shouldn't",
                    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}

    corpus = []
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english")) - custom_stopwords
    text = input_df["text"]

    for i in range(len(text)):
        review = re.sub('[^a-zA-Z]', ' ',text[i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if word not in stop_words]
        review = " ".join(review)
        corpus.append(review)
    return corpus

def vectorize(text):
    start = time.time()
    vectorizer.fit(text)
    x = vectorizer.transform(text)
    end = time.time()
    print(f"Vectorizing time: {end - start} sec.")
    return x.toarray()

def train_model(x, y):
    start = time.time()
    nb = MultinomialNB()
    nb.fit(x, y)
    print("Model trained")
    end = time.time()
    print(f"Model training time: {end - start} sec.")
    return nb
    
def dump_pickles(nb, vectorizer):
    joblib.dump(nb, "yelpmodel.pkl")
    joblib.dump(vectorizer, "yelp_vectorizer.pkl")

if __name__ == "__main__":
    print('Starting test')
    vectorizer = CountVectorizer(min_df=1)
    data = load_data()
    data = data.head(10000) #limiting to 10000 records for this project
    text = clean_text(data)
    X = vectorize(text)
    y=data["stars"].head(10000)
    trained_model = train_model(X, y)
    dump_pickles(trained_model, vectorizer)
    

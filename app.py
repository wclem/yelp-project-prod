import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(layout="wide")

scaler = StandardScaler()

st.title("Restaurant Rating Prediction App")

st.caption("This app predicts the stars your review will get.")

st.divider()

review_text = st.text_input("Type in your review here.")

btn_predict = st.button("Predict the review")

st.divider()

model = joblib.load("yelpmodel.pkl")
vectorizer = joblib.load("yelp_vectorizer.pkl")

# text preprocessing modules
from string import punctuation
 
# text preprocessing modules
from nltk.tokenize import word_tokenize
 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression

# function to clean the text
@st.cache_data
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
 
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers
 
    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])
 
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stopwords.words()]
        text = " ".join(text)
 
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
 
    # Return a list of words
    st.write(text)
    return text



def vctrz(text):
    # Using vectorizer previously loaded from pickle
    
    # call `transform` to convert text to a bag of words
    x = vectorizer.transform(text)

    # CountVectorizer uses a sparse array to save memory, but it's easier in this assignment to 
    # convert back to a "normal" numpy array
    x = x.toarray()
    return x


#print("Transformed text vector is \n{}".format(x))
if btn_predict:
    cleaned_text = [text_cleaning(review_text)]
    vector_text = vctrz(cleaned_text)
    prediction = model.predict(vector_text)
    st.write(prediction)
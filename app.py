import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import pickle




with open("svm_model_news.pkl","rb") as file:
    model=pickle.load(file)
tfidf_vector=TfidfVectorizer()

st.title("NEWS Classifier")

text=st.text_input("enter your text")

if text:
    preprocessed_text=tfidf_vector.fit_transform(text)
    prediction=model.predict(preprocessed_text)
    print(prediction[0])


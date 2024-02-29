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
    text.str.replace("[^a-zA-Z0-9]"," ")
    st.write(text)


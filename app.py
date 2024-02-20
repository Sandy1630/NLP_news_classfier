import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import pickle




with open("svm_model_news.pkl","rb") as file:
    model=pickle.load(file)
with open("vectorizer_model.pkl","rb") as f:
    tfid_vector=pickle.load(f)
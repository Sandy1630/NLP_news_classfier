import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import pickle
import nltk
import re
nltk.download("punkt","stopwords")
from nltk.corpus import stopwords
from nltk import word_tokenize
stop_word=stopwords.words("english")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


with open("svm_model_news.pkl","rb") as file:
    model=pickle.load(file)
tfidf_vector=TfidfVectorizer()

st.title("NEWS Classifier")

text=st.text_input("enter your text")
 
if text:
    text=re.sub("[^a-zA-Z0-9]"," ",text)
    text=text.lower()
    def remove_stopword(text):
        paragraph_tokenized=word_tokenize(text)
        rev_new=" ".join([i for i in paragraph_tokenized if i not in stop_word])
        return rev_new
    text=remove_stopword(text)
    
    
    
    lemmatizer=WordNetLemmatizer()


    def nltk_tag_to_wordnet_tag(nltk_tag):
        if nltk_tag.startswith("J"):
            return wordnet.ADJ
        elif nltk_tag.startswith("V"):
            return wordnet.VERB
        if nltk_tag.startswith("N"):
            return wordnet.NOUN
        if nltk_tag.startswith("R"):
            return wordnet.ADV
        else:
            return None

    def lemmatize_sentence(sentence):
        nltk_tagged=nltk.pos_tag(nltk.word_tokenize(sentence))
    
        wordnet_tagged=map(lambda x: (x[0],nltk_tag_to_wordnet_tag(x[1])),nltk_tagged)
    
        lemmatize_sentence=[]
        for word,tag in wordnet_tagged:
            if tag is None:
                lemmatize_sentence.append(word)
            else:
                lemmatize_sentence.append(lemmatizer.lemmatize(word,tag))
        return " ".join(lemmatize_sentence)
    text=lemmatize_sentence(text)
    with open("C:\\Users\\santh\\vectorizer_model.pkl","rb") as file:
        tfidf=pickle.load(file)
        
    text=tfidf.transform([text])
    
    feature_names=tfidf.get_feature_names_out()

    with open("C:\\Users\\santh\\news_classifier\\NLP_news_classfier\\svm_model_news.pkl","rb") as f:
        model=pickle.load(f)
    
    lables={0:"Politics/Entertainment/War",1:"Photography",2:"food"}

    if st.button("predict"):
        result=model.predict(text)
        predict_lable=lables[result[0]]
        st.write(predict_lable)
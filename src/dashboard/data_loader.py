import pandas as pd
import pickle
from bertopic import BERTopic
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/reddit_data_final.csv")
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    return df

@st.cache_resource
def load_topic_model():
    with open("models/topic_model/topic_model.pkl", "rb") as f:
        return pickle.load(f)

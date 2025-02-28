import pandas as pd
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import pickle
import os

def create_document_texts(df):
    """Create document texts by combining title and selftext"""
    docs = []
    for idx, row in df.iterrows():
        title = row['clean_title'] if pd.notna(row['clean_title']) else ""
        selftext = row['clean_selftext'] if pd.notna(row['clean_selftext']) else ""
        docs.append(f"{title} {selftext}".strip())
    return docs

def train_topic_model(docs, n_topics=15, model_name="all-MiniLM-L6-v2"):
    """Train a BERTopic model on the documents"""
    # Initialize sentence transformer model
    sentence_model = SentenceTransformer(model_name)
    
    # Initialize BERTopic
    vectorizer = CountVectorizer(stop_words="english")
    topic_model = BERTopic(
        embedding_model=sentence_model,
        vectorizer_model=vectorizer,
        nr_topics=n_topics,
        calculate_probabilities=True
    )
    
    # Fit the model
    topics, probs = topic_model.fit_transform(docs)
    
    return topic_model, topics, probs

def save_topic_model(model, topics, probs, output_dir):
    """Save the topic model and results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    with open(os.path.join(output_dir, 'topic_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    # Save the topics and probabilities
    np.save(os.path.join(output_dir, 'topics.npy'), topics)
    np.save(os.path.join(output_dir, 'probs.npy'), probs)
    
    print(f"Topic model saved to {output_dir}")

def add_topics_to_dataframe(df, topics, probs):
    """Add topic assignments and probabilities to the dataframe"""
    df['topic'] = topics
    
    # Add top topic probability
    df['topic_probability'] = [prob.max() if len(prob) > 0 else 0 for prob in probs]
    
    return df

def run_topic_modeling(input_file, output_dir, output_file):
    """Run the complete topic modeling pipeline"""
    print("Starting topic modeling...")
    
    # Load processed data
    df = pd.read_csv(input_file)
    
    # Create document texts
    docs = create_document_texts(df)
    
    # Train topic model
    topic_model, topics, probs = train_topic_model(docs)
    
    # Save the model
    save_topic_model(topic_model, topics, probs, output_dir)
    
    # Add topics to dataframe
    df = add_topics_to_dataframe(df, topics, probs)
    
    # Save enhanced dataframe
    df.to_csv(output_file, index=False)
    print(f"Topic modeling complete. Enhanced data saved to {output_file}")

if __name__ == "__main__":
    input_file = "data/processed/cleaned_reddit_data.csv"
    output_dir = "models/topic_model"
    output_file = "data/processed/reddit_data_with_topics.csv"
    run_topic_modeling(input_file, output_dir, output_file)
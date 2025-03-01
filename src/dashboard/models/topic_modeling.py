import pandas as pd
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import pickle
import os
from umap import UMAP
from hdbscan import HDBSCAN

def create_document_texts(df):
    """Create document texts by combining title and selftext"""
    docs = []
    for _, row in df.iterrows():
        title = row.get('clean_title', '')
        selftext = row.get('clean_selftext', '')
        title = title if pd.notna(title) else ""
        selftext = selftext if pd.notna(selftext) else ""
        docs.append(f"{title} {selftext}".strip())
    return docs

def generate_topic_labels(topic_model, docs):
    """Generate meaningful topic names using representative documents and keywords"""
    topic_info = topic_model.get_topic_info()
    topic_labels = {}

    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:
            topic_labels[topic_id] = "Miscellaneous"
            continue
        
        # Get keywords for the topic
        top_words = [word for word, _ in topic_model.get_topic(topic_id)][:5]
        
        # Create shorter, more readable topic label - use just the keywords
        # without adding the parenthetical text from the representative document
        topic_name = " & ".join(top_words[:3]).title()
        
        # Remove the following lines that add the parenthetical preview:
        # rep_docs = topic_model.get_representative_docs(topic_id)
        # rep_doc_text = rep_docs[0] if rep_docs else "No example available"
        # preview = rep_doc_text[:30].replace("\n", " ").strip()
        # if len(preview) > 0:
        #     topic_name = f"{topic_name} ({preview}...)"
            
        topic_labels[topic_id] = topic_name

    return topic_labels

def train_topic_model(docs, model_name="all-MiniLM-L6-v2"):
    """Train BERTopic model with improved clustering components"""
    # Use a more advanced embedding model if needed
    sentence_model = SentenceTransformer(model_name)
    
    # Use n-grams up to 3 to capture more complex phrases
    vectorizer = CountVectorizer(ngram_range=(1, 3), 
                                stop_words="english", 
                                min_df=5,   # Ignore terms that appear in fewer than 5 documents
                                max_df=0.7) # Ignore terms that appear in more than 70% of documents
    
    # Use UMAP for dimensionality reduction with parameters tuned for better clustering
    umap_model = UMAP(n_neighbors=15, 
                     n_components=5,
                     min_dist=0.0, 
                     metric='cosine', 
                     random_state=42)
    
    # Use HDBSCAN for clustering with parameters for more coherent topics
    hdbscan_model = HDBSCAN(min_cluster_size=20,
                          min_samples=5,
                          prediction_data=True,
                          alpha=1.0,
                          cluster_selection_method='eom')
    
    # Create BERTopic model with custom components
    topic_model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics="auto",
        min_topic_size=20,  # Smaller min topic size for more granular topics
        calculate_probabilities=True,
        verbose=True
    )
    
    # Fit the model
    topics, probs = topic_model.fit_transform(docs)
    
    # Generate meaningful topic labels
    topic_labels = generate_topic_labels(topic_model, docs)
    
    return topic_model, topics, probs, topic_labels

def save_topic_model(model, topics, probs, topic_labels, output_dir):
    """Save the topic model and results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and data
    with open(os.path.join(output_dir, 'topic_model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    np.save(os.path.join(output_dir, 'topics.npy'), topics)
    np.save(os.path.join(output_dir, 'probs.npy'), probs)

    with open(os.path.join(output_dir, 'topic_labels.pkl'), 'wb') as f:
        pickle.dump(topic_labels, f)
    
    # Save topic info dataframe for later visualization
    topic_info = model.get_topic_info()
    topic_info.to_csv(os.path.join(output_dir, 'topic_info.csv'), index=False)
    
    # Save topic words for later visualization
    topic_words = {}
    for topic_id in set(topics):
        if topic_id != -1:  # Skip outlier topic
            words_scores = model.get_topic(topic_id)
            topic_words[topic_id] = words_scores
    
    with open(os.path.join(output_dir, 'topic_words.pkl'), 'wb') as f:
        pickle.dump(topic_words, f)
    
    print(f"Topic model saved to {output_dir}")

def add_topics_to_dataframe(df, topics, probs, topic_labels):
    """Add topic assignments and probabilities to the dataframe with additional metrics"""
    df['topic'] = topics
    df['topic_name'] = df['topic'].map(lambda x: topic_labels.get(x, f"Topic {x}"))
    
    # Add topic probability
    df['topic_probability'] = [max(prob) if isinstance(prob, np.ndarray) and len(prob) > 0 else 0 for prob in probs]
    
    # Add top 3 alternative topics for each document
    top_3_topics = []
    top_3_probs = []
    
    for prob in probs:
        if isinstance(prob, np.ndarray) and len(prob) > 0:
            # Get indices of top 3 probabilities
            top_idx = np.argsort(prob)[-3:][::-1]
            # Get corresponding topic IDs and probabilities
            topics_arr = np.array(list(topic_labels.keys()))
            top_topic = topics_arr[top_idx].tolist() if len(topics_arr) > 0 else []
            top_prob = prob[top_idx].tolist() if len(top_idx) > 0 else []
            
            top_3_topics.append(top_topic)
            top_3_probs.append(top_prob)
        else:
            top_3_topics.append([])
            top_3_probs.append([])
    
    df['alternative_topics'] = top_3_topics
    df['alternative_probabilities'] = top_3_probs
    
    # Clean up and organize
    df = df.sort_values(by='topic_probability', ascending=False)
    
    return df

def run_topic_modeling(input_file, output_dir, output_file, embedding_model="all-MiniLM-L6-v2"):
    """Execute the enhanced topic modeling pipeline"""
    print("Starting enhanced topic modeling...")

    df = pd.read_csv(input_file)
    
    # Create document texts
    docs = create_document_texts(df)
    print(f"Processing {len(docs)} documents")
    
    # Train topic model
    topic_model, topics, probs, topic_labels = train_topic_model(docs, model_name=embedding_model)
    
    # Save model and data for later visualization
    save_topic_model(topic_model, topics, probs, topic_labels, output_dir)
    
    # Add topics to dataframe
    df = add_topics_to_dataframe(df, topics, probs, topic_labels)
    
    # Get topic distribution for later visualization
    topic_counts = df['topic'].value_counts().reset_index()
    topic_counts.columns = ['topic', 'count']
    topic_counts['topic_name'] = topic_counts['topic'].map(lambda x: df.loc[df['topic'] == x, 'topic_name'].iloc[0] if any(df['topic'] == x) else f"Topic {x}")
    topic_counts.to_csv(os.path.join(output_dir, 'topic_counts.csv'), index=False)
    
    # Save enhanced dataframe
    df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"Topic modeling complete. Found {len(topic_labels) - 1} topics.")
    print(f"Top 5 topics by document count:")
    for _, row in topic_counts.head(5).iterrows():
        print(f"  - {row['topic_name']} ({row['count']} documents)")
    
    print(f"Enhanced data saved to {output_file}")
    print(f"Topic model and data saved to {output_dir} for later visualization")

if __name__ == "__main__":
    input_file = "data/processed/cleaned_reddit_data.csv"
    output_dir = "models/topic_model"
    output_file = "data/processed/reddit_data_with_topics.csv"
    
    # Use a more powerful embedding model for better topic separation
    # Options: "all-MiniLM-L6-v2" (fast), "all-mpnet-base-v2" (more accurate)
    embedding_model = "all-MiniLM-L6-v2"
    
    run_topic_modeling(input_file, output_dir, output_file, embedding_model)
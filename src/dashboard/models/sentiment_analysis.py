import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
import os
import pickle

# Download necessary NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

def analyze_sentiment(df, text_columns=['clean_title', 'clean_selftext']):
    """
    Add sentiment analysis scores to the dataframe, considering both title and selftext.
    Enhanced to better integrate with topic modeling data.
    """
    print("Starting sentiment analysis...")
    
    sid = SentimentIntensityAnalyzer()

    # Function to compute sentiment for a single text
    def get_sentiment(text):
        if pd.isna(text) or text.strip() == "":
            return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
        return sid.polarity_scores(text)
    
    # Analyze sentiment for each text column
    for col in text_columns:
        if col in df.columns:
            print(f"Analyzing sentiment for {col}...")
            sentiments = df[col].apply(get_sentiment)
            df[f'{col}_neg'] = sentiments.apply(lambda x: x['neg'])
            df[f'{col}_neu'] = sentiments.apply(lambda x: x['neu'])
            df[f'{col}_pos'] = sentiments.apply(lambda x: x['pos'])
            df[f'{col}_compound'] = sentiments.apply(lambda x: x['compound'])

    # Ensure required columns exist and handle missing values
    df['clean_title_compound'] = df.get('clean_title_compound', 0)
    df['clean_selftext_compound'] = df.get('clean_selftext_compound', 0)
    
    # Fill missing topics appropriately
    df['topic'] = df['topic'].fillna(-1).astype(int)
    df['topic_name'] = df['topic_name'].fillna("Miscellaneous")
    
    # Handle missing selftext and calculate content length
    df['has_selftext'] = ~df['clean_selftext'].isna() & (df['clean_selftext'].str.strip() != "")
    df['content_length'] = df['clean_selftext'].fillna("").str.len()
    df['title_length'] = df['clean_title'].fillna("").str.len()
    df['total_length'] = df['content_length'] + df['title_length']

    # Improved sentiment score calculation with dynamic weighting based on content
    def calculate_sentiment_score(row):
        if pd.isna(row.get('clean_selftext')) or row['content_length'] == 0:
            # If no selftext, use only title sentiment
            return row.get('clean_title_compound', 0)
        elif pd.isna(row.get('clean_title')) or row['title_length'] == 0:
            # If no title, use only selftext sentiment
            return row.get('clean_selftext_compound', 0)
        else:
            # Dynamic weighting based on relative lengths
            title_weight = 0.4
            selftext_weight = 0.6
            
            # If selftext is very short, give more weight to title
            if row['content_length'] < 50 and row['title_length'] > 0:
                title_weight = 0.7
                selftext_weight = 0.3
                
            # If title is very expressive (has strong sentiment), increase its weight
            title_sentiment = abs(row.get('clean_title_compound', 0))
            if title_sentiment > 0.5:
                title_weight = min(0.8, title_weight + 0.2)
                selftext_weight = 1 - title_weight
                
            return (title_weight * row.get('clean_title_compound', 0) + 
                    selftext_weight * row.get('clean_selftext_compound', 0))
    
    df['sentiment_score'] = df.apply(calculate_sentiment_score, axis=1)
    
    # Keep NaNs to prevent incorrect neutral classification
    df['sentiment_score'] = df['sentiment_score'].fillna(np.nan)

    # Classify sentiment into 5 categories for more nuanced analysis
    df['sentiment_category'] = pd.cut(
        df['sentiment_score'], 
        bins=[-1, -0.6, -0.2, 0.2, 0.6, 1], 
        labels=['very negative', 'negative', 'neutral', 'positive', 'very positive']
    )
    
    # For compatibility with older code, add a simpler 3-category classification
    df['sentiment_simple'] = pd.cut(
        df['sentiment_score'], 
        bins=[-1, -0.25, 0.25, 1], 
        labels=['negative', 'neutral', 'positive']
    )

    print("Sentiment analysis complete.")
    return df

def analyze_topic_sentiment(df, output_dir):
    """
    Analyze sentiment patterns within topics and save the results
    for later visualization.
    """
    print("Analyzing sentiment patterns by topic...")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Define a named function for sentiment value counts
    def get_sentiment_counts(x):
        return x.value_counts().to_dict()
    
    # Calculate sentiment statistics by topic
    topic_sentiment = df.groupby(['topic', 'topic_name']).agg({
        'sentiment_score': ['mean', 'median', 'std', 'count'],
        'sentiment_category': get_sentiment_counts,  # Use named function instead of lambda
        'clean_title_compound': 'mean',
        'clean_selftext_compound': 'mean'
    }).reset_index()
    
    # Flatten the multi-level columns
    topic_sentiment.columns = ['_'.join(col).strip('_') for col in topic_sentiment.columns.values]
    
    # Now the column will be named 'sentiment_category_get_sentiment_counts'
    sentiment_col = 'sentiment_category_get_sentiment_counts'
    
    # Calculate percentage of each sentiment category within each topic
    for topic_idx, row in topic_sentiment.iterrows():
        sentiment_counts = row[sentiment_col]
        total = sum(sentiment_counts.values())
        sentiment_percentages = {f"{category}_pct": count/total*100 
                                  for category, count in sentiment_counts.items()}
        
        # Add percentage columns to the dataframe
        for category, percentage in sentiment_percentages.items():
            topic_sentiment.at[topic_idx, category] = percentage
    
    # Save topic sentiment data
    topic_sentiment.to_csv(os.path.join(output_dir, 'topic_sentiment.csv'), index=False)
    
    # Create a pivot table of topics vs sentiment categories
    sentiment_pivot = pd.pivot_table(
        df, 
        values='id', 
        index=['topic', 'topic_name'],
        columns='sentiment_category',
        aggfunc='count', 
        fill_value=0
    )
    
    # Save pivot table for visualization
    sentiment_pivot.to_csv(os.path.join(output_dir, 'topic_sentiment_pivot.csv'))
    
    # Create a normalized version (percentages)
    sentiment_pivot_pct = sentiment_pivot.div(sentiment_pivot.sum(axis=1), axis=0) * 100
    sentiment_pivot_pct.to_csv(os.path.join(output_dir, 'topic_sentiment_pivot_pct.csv'))
    
    print(f"Topic sentiment analysis complete. Results saved to {output_dir}")
    return topic_sentiment
def extract_sentiment_keywords(df, output_dir):
    """
    Extract keywords associated with positive and negative sentiment
    within each topic to understand emotion drivers.
    """
    print("Extracting sentiment-associated keywords by topic...")
    
    def get_sentiment_words(texts, sentiment_threshold):
        """Extract words strongly associated with a sentiment"""
        word_sentiments = {}
        
        for text, sentiment in texts:
            if pd.isna(text) or abs(sentiment) < abs(sentiment_threshold):
                continue
                
            # Tokenize the text
            try:
                words = word_tokenize(text.lower())
                # Add sentiment association to each word
                for word in words:
                    if len(word) > 3:  # Skip very short words
                        if word in word_sentiments:
                            word_sentiments[word].append(sentiment)
                        else:
                            word_sentiments[word] = [sentiment]
            except:
                continue
        
        # Calculate average sentiment for each word
        word_avg_sentiment = {word: np.mean(sentiments) 
                              for word, sentiments in word_sentiments.items() 
                              if len(sentiments) >= 3}  # Only words appearing multiple times
        
        return word_avg_sentiment
    
    # Process each topic to find sentiment-associated words
    sentiment_keywords = {}
    
    for topic_id in df['topic'].unique():
        topic_df = df[df['topic'] == topic_id]
        
        # Skip topics with very few documents
        if len(topic_df) < 10:
            continue
            
        topic_name = topic_df['topic_name'].iloc[0]
        
        # Combine title and selftext with their sentiment scores
        text_sentiments = []
        for _, row in topic_df.iterrows():
            if pd.notna(row.get('clean_title')):
                text_sentiments.append((row['clean_title'], row['clean_title_compound']))
            if pd.notna(row.get('clean_selftext')):
                text_sentiments.append((row['clean_selftext'], row['clean_selftext_compound']))
        
        # Get sentiment-associated words
        word_sentiments = get_sentiment_words(text_sentiments, 0.3)
        
        if word_sentiments:
            # Sort by sentiment value
            sorted_words = sorted(word_sentiments.items(), key=lambda x: x[1])
            
            # Get most negative and most positive words
            negative_words = [(word, score) for word, score in sorted_words if score < -0.3][:20]
            positive_words = [(word, score) for word, score in sorted_words if score > 0.3][-20:]
            
            sentiment_keywords[topic_id] = {
                'topic_name': topic_name,
                'negative_words': negative_words,
                'positive_words': positive_words
            }
    
    # Save sentiment keywords
    with open(os.path.join(output_dir, 'sentiment_keywords.pkl'), 'wb') as f:
        pickle.dump(sentiment_keywords, f)
    
    print(f"Sentiment keyword extraction complete. Results saved to {output_dir}")
    
    return sentiment_keywords

def run_sentiment_analysis(input_file, output_file, output_dir="models/sentiment_analysis"):
    """
    Run the enhanced sentiment analysis pipeline and save results.
    Now integrates with topic modeling data for richer analysis.
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Ensure we have an 'id' column for aggregation
    if 'id' not in df.columns:
        df['id'] = df.index
    
    # Run basic sentiment analysis
    df = analyze_sentiment(df)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze sentiment patterns by topic
    topic_sentiment = analyze_topic_sentiment(df, output_dir)
    
    # Extract sentiment-associated keywords for each topic
    sentiment_keywords = extract_sentiment_keywords(df, output_dir)
    
    # Calculate overall sentiment statistics
    sentiment_stats = {
        'overall_mean': df['sentiment_score'].mean(),
        'overall_median': df['sentiment_score'].median(),
        'sentiment_counts': df['sentiment_category'].value_counts().to_dict(),
        'topic_count': len(df['topic'].unique()),
        'document_count': len(df)
    }
    
    # Save stats for later use
    with open(os.path.join(output_dir, 'sentiment_stats.pkl'), 'wb') as f:
        pickle.dump(sentiment_stats, f)
    
    # Save the enhanced dataframe
    df.to_csv(output_file, index=False)
    print(f"Sentiment-enhanced data saved to {output_file}")
    print(f"Additional analysis data saved to {output_dir}")

if __name__ == "__main__":
    input_file = "data/processed/reddit_data_with_topics.csv"
    output_file = "data/processed/reddit_data_final.csv"
    output_dir = "models/sentiment_analysis"
    run_sentiment_analysis(input_file, output_file, output_dir)
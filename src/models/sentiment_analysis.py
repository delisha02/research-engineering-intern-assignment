import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

def analyze_sentiment(df, text_columns=['clean_title', 'clean_selftext']):
    """
    Add sentiment analysis scores to the dataframe
    """
    print("Starting sentiment analysis...")
    
    # Initialize the sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    
    # Function to compute sentiment for a single text
    def get_sentiment(text):
        if pd.isna(text) or text == "":
            return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
        return sid.polarity_scores(text)
    
    # Analyze sentiment for each text column
    for col in text_columns:
        if col in df.columns:
            print(f"Analyzing sentiment for {col}...")
            
            # Apply sentiment analysis
            sentiments = df[col].apply(get_sentiment)
            
            # Extract sentiment components
            df[f'{col}_neg'] = sentiments.apply(lambda x: x['neg'])
            df[f'{col}_neu'] = sentiments.apply(lambda x: x['neu'])
            df[f'{col}_pos'] = sentiments.apply(lambda x: x['pos'])
            df[f'{col}_compound'] = sentiments.apply(lambda x: x['compound'])
    
    # Create combined sentiment score (average of title and content if both exist)
    #if all(f'clean_title_compound' in df.columns and f'clean_selftext_compound' in df.columns):
    if all([f'clean_title_compound' in df.columns, f'clean_selftext_compound' in df.columns]):

        # Weight title more heavily if selftext is empty
        df['content_length'] = df['clean_selftext'].str.len()
        
        # Calculate weighted compound score
        conditions = [
            df['content_length'] == 0,  # Only title
            df['content_length'] > 0    # Both title and content
        ]
        choices = [
            df['clean_title_compound'],  # Only title
            0.4 * df['clean_title_compound'] + 0.6 * df['clean_selftext_compound']  # Both
        ]
        df['sentiment_score'] = np.select(conditions, choices, default=0)
        
        # Add sentiment category
        df['sentiment_category'] = pd.cut(
            df['sentiment_score'], 
            bins=[-1, -0.25, 0.25, 1], 
            labels=['negative', 'neutral', 'positive']
        )
    
    print("Sentiment analysis complete.")
    return df

def run_sentiment_analysis(input_file, output_file):
    """Run the complete sentiment analysis pipeline"""
    # Load data
    df = pd.read_csv(input_file)
    
    # Analyze sentiment
    df = analyze_sentiment(df)
    
    # Save enhanced dataframe
    df.to_csv(output_file, index=False)
    print(f"Enhanced data with sentiment saved to {output_file}")

if __name__ == "__main__":
    input_file = "data/processed/reddit_data_with_topics.csv"
    output_file = "data/processed/reddit_data_final.csv"
    run_sentiment_analysis(input_file, output_file)
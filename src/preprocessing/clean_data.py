import pandas as pd
import re
import json
from datetime import datetime
import nltk
from nltk.corpus import stopwords

# Load the raw dataset
"""Load and flatten Reddit JSON data into a pandas DataFrame."""
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                raw_entry = json.loads(line)  # Load each line as a JSON object
                if "data" in raw_entry:  # Extract nested 'data' field
                    data.append(raw_entry["data"])
                else:
                    data.append(raw_entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON line. Error: {e}")

    df = pd.DataFrame(data)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

    # Check for missing columns and add them if necessary
    for col in ['title', 'selftext', 'created_utc']:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found, adding as empty column.")
            df[col] = "" if col != 'created_utc' else pd.NaT  # NaT for timestamps

    print("First few rows of dataset:")
    print(df[['title', 'selftext', 'created_utc']].head())  # Show relevant columns

    return df

# Define expanded misinformation-related keywords
def get_expanded_keywords():
    """Return expanded list of misinformation-related keywords"""
    return [
        "trump", "biden", "election", "vote", "fraud", "fake news", "conspiracy", 
        "democrat", "republican", "deep state", "hoax", "rigged", "stolen election",
        "media bias", "voter suppression", "mail-in ballots", "stop the steal",
        "ballot harvesting", "voting machine", "dominion", "electoral college",
        "constitutional crisis", "smartmatic", "antrim county", "sidney powell",
        "giuliani", "kraken", "audit", "recount", "election integrity", "q anon",
        "censorship", "big tech", "section 230", "hunter biden", "laptop", 
        "disinformation", "propaganda", "fact check", "debunked", "swing state"
    ]

# Filter data based on keywords
def filter_by_keywords(df, keywords):
    """Filter the DataFrame to include only posts containing keywords"""
    if df.empty:
        print("Warning: Empty DataFrame, skipping keyword filtering.")
        return df

    keywords = [k.lower() for k in keywords]
    pattern = '|'.join(r'\b{}\b'.format(re.escape(k)) for k in keywords)
    
    df['title'] = df['title'].fillna("")
    df['selftext'] = df['selftext'].fillna("")
    
    title_mask = df['title'].str.lower().str.contains(pattern, na=False)
    selftext_mask = df['selftext'].str.lower().str.contains(pattern, na=False)
    combined_mask = title_mask | selftext_mask
    
    filtered_df = df[combined_mask].copy()
    print(f"Filtered dataset contains {len(filtered_df)} rows")
    
    return filtered_df

# Clean text data
def clean_text(df):
    """Clean text fields in the DataFrame"""
    if df.empty:
        print("Warning: Empty DataFrame, skipping text cleaning.")
        return df

    def clean(text):
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['clean_title'] = df['title'].apply(clean)
    df['clean_selftext'] = df['selftext'].apply(clean)

    return df

# Convert timestamps
def convert_timestamps(df):
    """Convert Unix timestamps to datetime format"""
    if df.empty or 'created_utc' not in df.columns:
        print("Warning: Empty DataFrame or missing 'created_utc', skipping timestamp conversion.")
        return df

    try:
        df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')
    except Exception as e:
        print(f"Warning: Error converting timestamps - {e}")

    df['year'] = df['created_utc'].dt.year
    df['month'] = df['created_utc'].dt.month
    df['day'] = df['created_utc'].dt.day
    df['hour'] = df['created_utc'].dt.hour
    df['weekday'] = df['created_utc'].dt.weekday

    return df

# Add engagement metrics
def add_engagement_metrics(df):
    """Add derived engagement metrics"""
    if df.empty or not all(col in df.columns for col in ['score', 'num_comments']):
        print("Warning: Empty DataFrame or missing engagement columns, skipping engagement metrics.")
        return df

    try:
        score_norm = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min() + 1)
        comment_norm = (df['num_comments'] - df['num_comments'].min()) / (df['num_comments'].max() - df['num_comments'].min() + 1)
        df['engagement_index'] = (0.5 * score_norm + 0.5 * comment_norm) * 100
    except Exception as e:
        print(f"Warning: Error calculating engagement metrics - {e}")

    return df

# Main preprocessing function
def preprocess_data(input_file, output_file):
    """Run the complete preprocessing pipeline"""
    print("Starting data preprocessing...")
    
    # Check if NLTK data needs to be downloaded (only needed for imports, not actual use in current script)
    try:
        stopwords.words('english')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
    
    df = load_data(input_file)
    
    if df.empty:
        print("Error: No valid data found in the input file.")
        return None

    keywords = get_expanded_keywords()
    df = filter_by_keywords(df, keywords)

    if df.empty:
        print("Warning: No posts matched the keyword filter. Skipping further processing.")
    else:
        df = clean_text(df)
        df = convert_timestamps(df)
        df = add_engagement_metrics(df)

    df.to_csv(output_file, index=False)
    print(f"Preprocessing complete. Processed data saved to {output_file}")
    print(f"Final dataset shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    input_file = "data/raw/data.jsonl"
    output_file = "data/processed/cleaned_reddit_data.csv"
    preprocess_data(input_file, output_file)
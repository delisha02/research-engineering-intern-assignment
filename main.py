import os
import argparse
import subprocess
import sys
import time

def run_command(command, description):
    """Run a shell command with a description"""
    print(f"\n--- {description} ---")
    start_time = time.time()
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        sys.exit(1)
    print(f"Completed in {time.time() - start_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description="Run the Reddit misinformation dashboard pipeline")
    parser.add_argument('--skip-clean', action='store_true', help='Skip data cleaning step')
    parser.add_argument('--skip-model', action='store_true', help='Skip model training steps')
    parser.add_argument('--data-file', type=str, default='data/raw/data.jsonl', 
                        help='Path to the raw Reddit data file')
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models/topic_model', exist_ok=True)
    
    # Check if data exists
    if not os.path.exists(args.data_file):
        print(f"Error: Input data file {args.data_file} not found")
        sys.exit(1)

        # Step 1: Clean and preprocess the data
    if not args.skip_clean:
        run_command(
            f"python src/preprocessing/clean_data.py --input {args.data_file} --output data/processed/cleaned_reddit_data.csv",
            "Data Cleaning and Preprocessing"
        )
    
    # Step 2: Train topic model
    if not args.skip_model:
        run_command(
            "python src/models/topic_modeling.py --input data/processed/cleaned_reddit_data.csv --output data/processed/reddit_data_with_topics.csv",
            "Topic Modeling"
        )
        
        # Step 3: Run sentiment analysis
        run_command(
            "python src/models/sentiment_analysis.py --input data/processed/reddit_data_with_topics.csv --output data/processed/reddit_data_final.csv",
            "Sentiment Analysis"
        )
    
    # Step 4: Run the dashboard
    print("\n--- Starting Dashboard ---")
    run_command(
        "streamlit run src/dashboard/app.py",
        "Running Dashboard"
    )

if __name__ == "__main__":
    main()
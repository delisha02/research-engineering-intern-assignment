# research-engineering-intern-assignment
# 📊 The Political Spectrum: Reddit Communities & Their Political Sentiments  

## **Overview**  
This project is an interactive dashboard that visualizes political discussions on Reddit. It provides insights into how information narratives evolved, which communities were most active, and how users engaged with different types of content.  

---

## **Features**  
✅ **Interactive Dashboard** – Built with **Streamlit** to explore Reddit data dynamically.  
✅ **Sentiment Analysis** – Uses **VADER** to classify posts as **positive, neutral, or negative**.  
✅ **Topic Modeling** – Implements **BERTopic** to identify key discussion topics.  
✅ **Engagement Metrics** – Shows **post scores, comments, and subreddit activity**.  
✅ **Time-Series Analysis** – Tracks **sentiment and post frequency trends over time**.  

---

## **Project Structure**  
```bash
📂 reddit-political-dashboard/
├── app.py                # Main Streamlit dashboard  
├── data_loader.py        # Loads Reddit dataset & topic model  
├── sentiment_analysis.py # Analyzes sentiment scores  
├── topic_modeling.py     # Applies BERTopic to extract key topics  
├── clean_data.py         # Preprocesses raw Reddit data  
├── styles.css            # Custom CSS for UI enhancements  
├── data/                 # Folder for processed datasets  
│   ├── raw/              # Contains unprocessed Reddit data  
│   ├── processed/        # Contains cleaned and analyzed data  
├── models/               # Folder for trained topic models  
│   ├── topic_model/      # Stores topic modeling artifacts  
│   ├── sentiment_analysis/ # Stores sentiment analysis results  
└── README.md             # Documentation  
Installation & Setup
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/reddit-political-dashboard.git
cd reddit-political-dashboard
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Ensure the following Python libraries are installed:

Streamlit (for dashboard)
Pandas (for data processing)
Plotly (for interactive charts)
NLTK (for sentiment analysis)
BERTopic (for topic modeling)
3️⃣ Download NLTK Data (For Sentiment Analysis)
python
Copy
Edit
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
4️⃣ Run the Dashboard
bash
Copy
Edit
streamlit run app.py
How It Works
1️⃣ Data Processing
Raw Reddit data (.jsonl format) is loaded via clean_data.py.
Text is cleaned, filtered by political keywords, and timestamps converted.
Engagement metrics (score, comments) are computed.
2️⃣ Sentiment Analysis
Uses NLTK VADER to analyze title & selftext sentiment.
Classifies posts into 5 categories: Very Negative, Negative, Neutral, Positive, Very Positive.
Stores results in reddit_data_final.csv.
3️⃣ Topic Modeling
Uses BERTopic (based on sentence embeddings) to group posts into meaningful topics.
Extracts top 3 topics per post and stores probabilities.
4️⃣ Dashboard Visualization
Users can filter posts by date, subreddit, and sentiment.
Displays key metrics, time-series charts, and topic distributions.
Provides a post explorer with sorting options (most positive/negative, top comments, etc.).
Key Components
📌 app.py – The Streamlit Dashboard
Loads data & applies sidebar filters (date, subreddit).
Displays key statistics (Total Posts, Avg. Sentiment).
Renders interactive charts & graphs using Plotly.
🔍 sentiment_analysis.py – Sentiment Scoring
Uses NLTK VADER to compute sentiment.
Adjusts weights dynamically (title vs. selftext).
Classifies sentiment into 5 categories.
📢 topic_modeling.py – Topic Discovery
Trains BERTopic using UMAP & HDBSCAN clustering.
Extracts keywords & representative posts per topic.
Saves topic distributions for visualization.
🔄 data_loader.py – Loads Processed Data
Fetches reddit_data_final.csv.
Loads the trained topic model (.pkl file).
🎨 styles.css – UI Enhancements
Customizes header colors & sidebar styles.
Adjusts font size for better readability.
Requirements
Save the following dependencies into a requirements.txt file:

txt
Copy
Edit
# Data Processing
pandas>=1.5.0
numpy>=1.20.0
scikit-learn>=1.0.0

# Data Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0
streamlit>=1.20.0

# NLP & Topic Modeling
nltk>=3.7.0
bertopic>=0.13.0
sentence-transformers>=2.2.0
Future Enhancements
🚀 Real-time Reddit API integration for live updates.
🤖 More advanced NLP models (e.g., RoBERTa for sentiment).
📊 Dashboard customization with user-defined topic filters.
🔍 Named Entity Recognition (NER) for tracking politicians & policies.



License
📜 MIT License – Free to use & modify.

yaml
Copy
Edit

---
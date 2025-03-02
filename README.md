
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

---

## **Deployment**
Deployed using Streamlit Cloud.

[Reddit's Political Spectrum Analysis](https://reddit-political-spectrum.streamlit.app/)

---

## **Project Structure**  
```
📂 research-engineering-intern-assignment/
│
├── 📂 src/
│   ├── 📂 dashboard/
│   │   ├── app.py                # Main Streamlit dashboard  
│   │   ├── data_loader.py        # Loads Reddit dataset & topic model  
│   │   ├── 📂 static/             # Static assets (CSS, images, etc.)
│   │   │   ├── styles.css         # Custom CSS for UI enhancements  
│   │   ├── 📂 models/             # Machine learning models for dashboard  
│   │   │   ├── sentiment_analysis.py  # Sentiment scoring  
│   │   │   ├── topic_modeling.py      # Topic modeling with BERTopic  
│   │
│   ├── 📂 preprocessing/
│   │   ├── clean_data.py          # Data cleaning and preprocessing  
│
├── 📂 models/                     # Folder for trained models  
│   ├── 📂 topic_model/             # Trained BERTopic models and data  
│   │   ├── topic_model.pkl        # Trained BERTopic model  
│   │   ├── topics.npy             # Topic assignments per post  
│   │   ├── probs.npy              # Probability scores of topics  
│   │   ├── topic_labels.pkl       # Topic names generated from BERTopic  
│   │   ├── topic_words.pkl        # Top words per topic  
│   │   ├── topic_counts.csv       # Number of posts per topic  
│   │   ├── topic_info.csv         # Topic metadata for visualization  
│   │
│   ├── 📂 sentiment_analysis/      # Trained sentiment analysis models and data  
│   │   ├── topic_sentiment.csv         # Sentiment data per topic  
│   │   ├── topic_sentiment_pivot.csv   # Pivot table of topics vs sentiment  
│   │   ├── topic_sentiment_pivot_pct.csv  # Percentage-based topic sentiment  
│   │   ├── sentiment_stats.pkl          # Sentiment statistics  
│   │   ├── sentiment_keywords.pkl       # Keywords strongly associated with sentiment  
│
├── 📂 data/                       # Folder for datasets  
│   ├── 📂 raw/                    # Unprocessed Reddit data  
│   ├── 📂 processed/              # Cleaned and analyzed data  
│
├── requirements.txt                # Dependencies  
├── README.md                       # Documentation  

```

---

## **Installation & Setup**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/delisha02/research-engineering-intern-assignment.git
cd research-engineering-intern-assignment
```

### **2️⃣ Create Virtual Environment and  Install Dependencies**
```bash
#Create Virtual Environment
python -m venv venv
# Activate the virtual environment:
# On macOS and Linux:
source myenv/bin/activate
# On Windows
venv\Scripts\activate
#Install dependencies
pip install -r requirements.txt
``` 

### **3️⃣ Running the Scripts (from root directory: research-engineering-intern-assignment):**
```bash
# Data cleaning 
python src/preprocessing/clean_data.py
```
```bash
# Topic modeling
python src/dashboard/models/topic_modeling.py
```
```bash
# Sentiment analysis
python src/dashboard/models/sentiment_analysis.py

```

### **4️⃣ Run the Dashboard**  
```bash
streamlit run src/dashboard/app.py
```

---

## **How It Works**  

### **1️⃣ Data Processing (`src/preprocessing/clean_data.py`)**  
- Loads raw Reddit JSON data.  
- Cleans and filters posts based on political keywords.  
- Converts timestamps and computes engagement metrics.  
- Saves cleaned data to `data/processed/reddit_data_final.csv`.  

### **2️⃣ Sentiment Analysis (`src/dashboard/models/sentiment_analysis.py`)**  
- Uses **NLTK’s VADER** to compute sentiment for post titles & selftext.  
- Assigns sentiment categories (Very Negative, Negative, Neutral, Positive, Very Positive).  
- Saves sentiment-enhanced data.  

### **3️⃣ Topic Modeling (`src/dashboard/models/topic_modeling.py`)**  
- Uses **BERTopic** to extract discussion topics.  
- Assigns the top **3 topics per post** with confidence scores.  
- Stores topics for visualization in the dashboard.  

### **4️⃣ Interactive Dashboard (`src/dashboard/app.py`)**  
- Allows users to filter posts by **date, subreddit, and sentiment**.  
- Displays **key statistics, time-series trends, and topic distributions**.  
- Provides a **post explorer** for sorting by sentiment, comments, and engagement.  

---

## **Key Components**  

### 📌 **`src/dashboard/app.py`** – The Streamlit Dashboard  
- Loads data & applies sidebar filters (date, subreddit).  
- Displays **key statistics** (Total Posts, Avg. Sentiment).  
- Renders interactive **charts & graphs** using **Plotly**.  

### 🔍 **`src/dashboard/models/sentiment_analysis.py`** – Sentiment Scoring  
- Uses **NLTK VADER** to compute sentiment.  
- Adjusts **weights dynamically** (title vs. selftext).  
- Classifies sentiment into **5 categories**.  

### 📢 **`src/dashboard/models/topic_modeling.py`** – Topic Discovery  
- Trains **BERTopic** using **UMAP & HDBSCAN** clustering.  
- Extracts **keywords & representative posts** per topic.  
- Saves **topic distributions** for visualization.  

### 🔄 **`src/dashboard/data_loader.py`** – Loads Processed Data  
- Fetches `reddit_data_final.csv`.  
- Loads the trained **topic model (`.pkl` file)**.  

### 🎨 **`src/dashboard/static/styles.css`** – UI Enhancements  
- Customizes **header colors & sidebar styles**.  
- Adjusts **font size for better readability**.  

---

## **Dashboard Overview**
Here’s how the dashboard looks:

![Dashboard](images/Key%20Metrics.png)
![Sidebar to Filter](images/Filters.png)
![](images/Posts%20Activity%20Over%20Time.png)
![](images/Sentiments%20Over%20Time.png)
![](images/Top%2015%20Subreddits.png)
![](images/Top%2010%20Subreddits%20by%20Engagement.png)
![](images/Top%2010%20Discussion.png)
![](images/Overall%20Topic%20Distribution.png)
![](images/Explore%20Posts.png)
![](images/Explore%20posts%20(sorted%20by%20Positive).png)
---

## **Future Enhancements**  
🚀 **Real-time Reddit API integration** for live updates.  
🤖 **More advanced NLP models** (e.g., RoBERTa for sentiment).  
📊 **Dashboard customization** with user-defined topic filters.  
🔍 **Named Entity Recognition (NER)** for tracking politicians & policies.  

---
---

## **Contributer**  
Delisha Naik: delisha02
```

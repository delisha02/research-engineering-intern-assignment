import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
from pyvis.network import Network
import pickle
import os
import json
from bertopic import BERTopic
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

#os.environ["STREAMLIT_WATCHDOG_LOG_LEVEL"] = "error"  # Suppress watchdog errors

# Set page configuration
st.set_page_config(
    page_title="Reddit Misinformation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/reddit_data_final.csv")
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    return df

# Load topic model
@st.cache_resource
def load_topic_model():
    with open("models/topic_model/topic_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def load_css(css_file_path):
    """
    Load and apply CSS from an external file
    
    Parameters:
    css_file_path (str): Path to the CSS file
    """
      # Get the absolute path based on the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_css_path = os.path.join(script_dir, css_file_path)
    
    if os.path.exists(absolute_css_path):
        with open(absolute_css_path, "r") as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.error(f"CSS file not found: {absolute_css_path}")

# Main function
def main():
    # Load CSS
    load_css("static/styles.css")  

    # Add title and description
    st.title("📊 Reddit Election Misinformation Dashboard")
    st.markdown("""
    This dashboard visualizes patterns of election misinformation on Reddit.
    Explore how misinformation narratives evolved, which communities were most active,
    and how users engaged with different types of content.
    """)
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        topic_model = load_topic_model()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range filter
    min_date = df['created_utc'].min().date()
    max_date = df['created_utc'].max().date()
    
    start_date = st.sidebar.date_input("Start Date", min_date)
    end_date = st.sidebar.date_input("End Date", max_date)
    
    # Convert to datetime for filtering
    start_datetime = pd.Timestamp(start_date)
    end_datetime = pd.Timestamp(end_date) + timedelta(days=1) - timedelta(seconds=1)
    
    # Apply date filter
    filtered_df = df[(df['created_utc'] >= start_datetime) & (df['created_utc'] <= end_datetime)]
    
    # Subreddit filter
    top_subreddits = df['subreddit'].value_counts().head(15).index.tolist()
    selected_subreddits = st.sidebar.multiselect(
        "Select Subreddits",
        options=["All"] + top_subreddits,
        default=["All"]
    )
    
    if "All" not in selected_subreddits and selected_subreddits:
        filtered_df = filtered_df[filtered_df['subreddit'].isin(selected_subreddits)]
    
    # Topic filter
    topic_info = topic_model.get_topic_info()
    topic_dict = {row['Topic']: row['Name'] for _, row in topic_info.iterrows() if row['Topic'] != -1}
    topic_options = ["All"] + [f"Topic {k}: {v[:30]}..." for k, v in topic_dict.items()]
    
    selected_topics = st.sidebar.multiselect(
        "Select Topics",
        options=topic_options,
        default=["All"]
    )
    
    if "All" not in selected_topics and selected_topics:
        selected_topic_ids = [int(topic.split(":")[0].replace("Topic ", "")) for topic in selected_topics]
        filtered_df = filtered_df[filtered_df['topic'].isin(selected_topic_ids)]
    
    # Display metrics
    st.header("📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Posts", f"{len(filtered_df):,}")
    
    with col2:
        avg_score = filtered_df['score'].mean()
        st.metric("Avg. Post Score", f"{avg_score:.1f}")
    
    with col3:
        avg_comments = filtered_df['num_comments'].mean()
        st.metric("Avg. Comments", f"{avg_comments:.1f}")
    
    with col4:
        sentiment_avg = filtered_df['sentiment_score'].mean()
        sentiment_label = "Positive" if sentiment_avg > 0.05 else "Negative" if sentiment_avg < -0.05 else "Neutral"
        st.metric("Avg. Sentiment", f"{sentiment_label} ({sentiment_avg:.2f})")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Temporal Analysis", "Community Analysis", "Topic Analysis", "Content Explorer"])
    
    # Tab 1: Temporal Analysis
    with tab1:
        st.subheader("Post Activity Over Time")
        
        # Group by date and count posts
        time_df = filtered_df.groupby(filtered_df['created_utc'].dt.date).size().reset_index(name='count')
        time_df['created_utc'] = pd.to_datetime(time_df['created_utc'])
        
        # Create time series plot
        fig = px.line(
            time_df, 
            x='created_utc', 
            y='count',
            title="Number of Posts Over Time",
            labels={"created_utc": "Date", "count": "Number of Posts"}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment over time
        st.subheader("Sentiment Trends")
        
        sentiment_time = filtered_df.groupby(filtered_df['created_utc'].dt.date)['sentiment_score'].mean().reset_index()
        sentiment_time['created_utc'] = pd.to_datetime(sentiment_time['created_utc'])
        
        fig = px.line(
            sentiment_time,
            x='created_utc',
            y='sentiment_score',
            title="Average Sentiment Over Time",
            labels={"created_utc": "Date", "sentiment_score": "Sentiment Score"}
        )
        fig.update_layout(height=400)
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Community Analysis
    with tab2:
        st.subheader("Community Distribution")
        
        # Subreddit distribution
        subreddit_counts = filtered_df['subreddit'].value_counts().reset_index()
        subreddit_counts.columns = ['subreddit', 'count']
        subreddit_counts = subreddit_counts.head(15)
        
        fig = px.bar(
            subreddit_counts,
            x='count',
            y='subreddit',
            title="Top 15 Subreddits by Post Count",
            labels={"count": "Number of Posts", "subreddit": "Subreddit"},
            orientation='h'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Engagement by subreddit
        st.subheader("Engagement by Community")
        
        engagement_by_sub = filtered_df.groupby('subreddit').agg({
            'score': 'mean',
            'num_comments': 'mean',
            'upvote_ratio': 'mean',
            'engagement_index': 'mean'
        }).reset_index()
        
        engagement_by_sub = engagement_by_sub.sort_values('engagement_index', ascending=False).head(10)
        
        fig = px.bar(
            engagement_by_sub,
            x='engagement_index',
            y='subreddit',
            title="Top 10 Subreddits by Engagement",
            labels={"engagement_index": "Engagement Index", "subreddit": "Subreddit"},
            orientation='h'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Topic Analysis
    with tab3:
        st.subheader("Topic Distribution")
        
        # Get topic counts
        topic_counts = filtered_df['topic'].value_counts().reset_index()
        topic_counts.columns = ['topic', 'count']
        topic_counts = topic_counts[topic_counts['topic'] != -1]  # Remove outlier topic
        
        # Add topic names
        topic_counts['topic_name'] = topic_counts['topic'].apply(
            lambda x: topic_dict.get(x, f"Topic {x}")[:30] + "..." if x in topic_dict else f"Topic {x}"
        )
        
        topic_counts = topic_counts.sort_values('count', ascending=False).head(10)
        
        fig = px.bar(
            topic_counts,
            x='count',
            y='topic_name',
            title="Top 10 Topics",
            labels={"count": "Number of Posts", "topic_name": "Topic"},
            orientation='h'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Topic visualization
        st.subheader("Topic Visualization")
        
        # Check if we have enough data
        if len(filtered_df) > 100:
            # Create topic visualization using BERTopic
            st.markdown("#### Interactive Topic Map")
            st.markdown("This visualization shows how topics relate to each other. Closer topics are more semantically similar.")
            
            # Create topic visualization
            fig = topic_model.visualize_topics()
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data for topic visualization. Try expanding your filter criteria.")
    
    # Tab 4: Content Explorer
    with tab4:
        st.subheader("Explore Posts")
        
        # Sort by options
        sort_by = st.selectbox(
            "Sort by",
            options=["Most Recent", "Highest Score", "Most Comments", "Most Positive", "Most Negative"]
        )
        
        if sort_by == "Most Recent":
            sorted_df = filtered_df.sort_values('created_utc', ascending=False)
        elif sort_by == "Highest Score":
            sorted_df = filtered_df.sort_values('score', ascending=False)
        elif sort_by == "Most Comments":
            sorted_df = filtered_df.sort_values('num_comments', ascending=False)
        elif sort_by == "Most Positive":
            sorted_df = filtered_df.sort_values('sentiment_score', ascending=False)
        elif sort_by == "Most Negative":
            sorted_df = filtered_df.sort_values('sentiment_score', ascending=True)
        
        # Display posts
        for i, (_, row) in enumerate(sorted_df.head(10).iterrows()):
            with st.expander(f"{row['title']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Title:** {row['title']}")
                    # Replace the problematic line (line 267) with this safer version:
                    if isinstance(row['selftext'], str):
                        st.markdown(f"**Content:** {row['selftext'][:500]}..." if len(row['selftext']) > 500 else f"**Content:** {row['selftext']}")
                    else:
                        st.markdown("**Content:** No content available")
                    #st.markdown(f"**Content:** {row['selftext'][:500]}..." if len(row['selftext']) > 500 else f"**Content:** {row['selftext']}")
                    
                    # Show keywords if available
                    if 'keywords_in_title' in row and isinstance(row['keywords_in_title'], str):
                        try:
                            keywords = eval(row['keywords_in_title'])
                            if keywords:
                                st.markdown(f"**Keywords:** {', '.join(keywords)}")
                        except:
                            pass
                
                with col2:
                    st.markdown(f"**Subreddit:** r/{row['subreddit']}")
                    st.markdown(f"**Date:** {row['created_utc'].strftime('%Y-%m-%d')}")
                    st.markdown(f"**Score:** {row['score']}")
                    st.markdown(f"**Comments:** {row['num_comments']}")
                    
                    # Show sentiment
                    sentiment = row['sentiment_score']
                    sentiment_color = "green" if sentiment > 0.05 else "red" if sentiment < -0.05 else "gray"
                    st.markdown(f"**Sentiment:** <span style='color:{sentiment_color}'>{sentiment:.2f}</span>", unsafe_allow_html=True)
                    
                    # Show topic if available
                    if 'topic' in row:
                        topic_id = int(row['topic'])
                        topic_name = topic_dict.get(topic_id, f"Topic {topic_id}")
                        st.markdown(f"**Topic:** {topic_name[:50]}")
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard created for data visualization assignment*")

if __name__ == "__main__":
    main()
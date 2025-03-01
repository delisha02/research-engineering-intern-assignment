import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from data_loader import load_data, load_topic_model

# Page Configuration
st.set_page_config(
    page_title="Reddit Misinformation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
def load_css():
    with open("src/dashboard/static/styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load Data & Models
with st.spinner("Loading Data..."):
    df = load_data()
    topic_model = load_topic_model()

# Apply CSS
load_css()

# Sidebar Filters
st.sidebar.header("Filters")

min_date, max_date = df['created_utc'].min().date(), df['created_utc'].max().date()
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)
start_dt, end_dt = pd.Timestamp(start_date), pd.Timestamp(end_date) + timedelta(days=1)

filtered_df = df[(df['created_utc'] >= start_dt) & (df['created_utc'] <= end_dt)]

# Subreddit Filter
top_subreddits = df['subreddit'].value_counts().head(15).index.tolist()
selected_subreddits = st.sidebar.multiselect("Select Subreddits", ["All"] + top_subreddits, default=["All"])

if "All" not in selected_subreddits:
    filtered_df = filtered_df[filtered_df['subreddit'].isin(selected_subreddits)]

# Topic Filter
topic_info = topic_model.get_topic_info()
topic_dict = {row['Topic']: row['Name'] for _, row in topic_info.iterrows() if row['Topic'] != -1}
topic_options = ["All"] + [f"Topic {k}: {v[:30]}..." for k, v in topic_dict.items()]
selected_topics = st.sidebar.multiselect("Select Topics", topic_options, default=["All"])

if "All" not in selected_topics:
    topic_ids = [int(topic.split(":")[0].replace("Topic ", "")) for topic in selected_topics]
    filtered_df = filtered_df[filtered_df['topic'].isin(topic_ids)]

st.title("📊 Reddit Election Misinformation Dashboard")
st.markdown("""
    This dashboard visualizes patterns of election misinformation on Reddit.
    Explore how misinformation narratives evolved, which communities were most active,
    and how users engaged with different types of content.
    """)
# Key Metrics
st.header("📈 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Posts", f"{len(filtered_df):,}")
col2.metric("Avg. Post Score", f"{filtered_df['score'].mean():.1f}")
col3.metric("Avg. Comments", f"{filtered_df['num_comments'].mean():.1f}")
sentiment_avg = filtered_df['sentiment_score'].mean()
col4.metric("Avg. Sentiment", f"{'Positive' if sentiment_avg > 0.05 else 'Negative' if sentiment_avg < -0.05 else 'Neutral'} ({sentiment_avg:.2f})")

# Tabs for Visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Temporal Analysis", "Community Analysis", "Topic Analysis", "Content Explorer"])

# Temporal Analysis
with tab1:
    st.subheader("Post Activity Over Time")
    time_df = filtered_df.groupby(filtered_df['created_utc'].dt.date).size().reset_index(name='count')
    fig = px.line(time_df, x='created_utc', y='count', title="Posts Over Time", labels={"created_utc": "Date", "count": "Posts"})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Sentiment Trends")  
    sentiment_time = filtered_df.groupby(filtered_df['created_utc'].dt.date)['sentiment_score'].mean().reset_index()
    sentiment_time['created_utc'] = pd.to_datetime(sentiment_time['created_utc'])
        
    fig = px.line(sentiment_time,x='created_utc',y='sentiment_score',title="Average Sentiment Over Time",labels={"created_utc": "Date", "sentiment_score": "Sentiment Score"})
    fig.update_layout(height=400)
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

# Community Analysis
with tab2:
    st.subheader("Top 15 Subreddits by Post Count")
    subreddit_counts = filtered_df['subreddit'].value_counts().reset_index()
    subreddit_counts.columns = ['subreddit', 'count']

    # Pie chart for post count by subreddit
    fig = px.pie(
        subreddit_counts.head(15),
        names='subreddit',
        values='count',
        title="Distribution of Posts by Subreddit",
        hole=0.4,  # Creates a donut chart effect
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig, use_container_width=True)

    # Engagement by subreddit (Using Pie Chart)
    st.subheader("Engagement by Community")    
    engagement_by_sub = filtered_df.groupby('subreddit').agg({
            'score': 'mean',
            'num_comments': 'mean',
            'upvote_ratio': 'mean',
            'engagement_index': 'mean'
        }).reset_index()
        
    engagement_by_sub = engagement_by_sub.sort_values('engagement_index', ascending=False).head(10)

    # Pie chart for engagement index by subreddit
    fig = px.pie(
        engagement_by_sub,
        names='subreddit',
        values='engagement_index',
        title="Top 10 Subreddits by Engagement",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig, use_container_width=True)

# Topic Analysis
with tab3:
    st.subheader("Top 10 Topics")
    topic_counts = filtered_df['topic'].value_counts().reset_index()
    topic_counts['topic_name'] = topic_counts['topic'].apply(lambda x: topic_dict.get(x, f"Topic {x}")[:30] + "...")
    fig = px.bar(topic_counts.head(10), x='topic_name', y='count', orientation='v')
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

# Content Explorer
with tab4:
    st.subheader("Explore Posts")
        
    # Sort by options
    sort_by = st.selectbox("Sort by",
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

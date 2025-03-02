import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from data_loader import load_data, load_topic_model, clean_topic_name

# Page Configuration
st.set_page_config(
    page_title="Reddit Analysis Dashboard",
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


# Dashboard Title
st.title("📊 The Political Spectrum: Reddit Communities & Their Political Sentiments")
st.markdown("""
    This dashboard visualizes patterns of political discussion on Reddit. 
    Explore how information narratives evolved, which communities were most active, 
    and how users engaged with different types of content.
""")

# Key Metrics
st.header("📈 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Posts", f"{len(filtered_df):,}")
col2.metric("Avg. Post Score", f"{filtered_df['score'].mean():.1f}")
col3.metric("Avg. Comments", f"{filtered_df['num_comments'].mean():.1f}")

if 'sentiment_score' in filtered_df.columns:
    if filtered_df['sentiment_score'].notna().sum() > 0:  # Check if any valid scores exist
        sentiment_avg = filtered_df['sentiment_score'].mean()
        sentiment_label = "Positive" if sentiment_avg > 0.05 else "Negative" if sentiment_avg < -0.05 else "Neutral"
        col4.metric("Avg. Sentiment", f"{sentiment_label} ({sentiment_avg:.2f})")
    else:
        col4.metric("Avg. Sentiment", "No Sentiment Data")  # Prevents NaN display


# Tabs for Visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Temporal Analysis", "Community Analysis", "Topic Analysis", "Content Explorer"])

# Temporal Analysis   , labels={"created_utc": "Date", "count": "Posts"})
with tab1:
    st.subheader("🕑 Post Activity Over Time")
    time_df = filtered_df.groupby(filtered_df['created_utc'].dt.date).size().reset_index(name='count')
    fig = px.line(time_df, x='created_utc', y='count', title="Posts Over Time", labels={"created_utc": "Date", "count": "Posts"})
    st.plotly_chart(fig, use_container_width=True)
   
    st.subheader("📉 Sentiment Trends")
    if 'sentiment_score' in filtered_df.columns:
        sentiment_time = filtered_df.groupby(filtered_df['created_utc'].dt.date)['sentiment_score'].mean().reset_index()
        sentiment_time['created_utc'] = pd.to_datetime(sentiment_time['created_utc'])
            
        fig = px.line(sentiment_time, x='created_utc', y='sentiment_score', title="Average Sentiment Over Time", labels={"created_utc": "Date", "sentiment_score": "Sentiment Score"})
        fig.update_layout(height=400)
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

#Community Engagement
with tab2:
    # **Subheader for Top 15 Subreddits**
    st.subheader("Top 15 Subreddits by Post Count")

    if 'subreddit' in filtered_df.columns:
        subreddit_counts = filtered_df['subreddit'].value_counts().reset_index()
        subreddit_counts.columns = ['subreddit', 'count']

        # Capitalize subreddit names
        subreddit_counts['subreddit'] = subreddit_counts['subreddit'].str.title()

        # Pie chart for post distribution
        fig = px.pie(
            subreddit_counts.head(15),
            names='subreddit',
            values='count',
            title="📊 Distribution of Posts by Subreddit",
            hole=0.4,  # Donut chart effect
            color_discrete_sequence=px.colors.qualitative.Set3
        )
    
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No subreddit data available.")

    # **Subheader for Engagement Analysis**
    st.subheader("Engagement by Community")

    if {'subreddit', 'engagement_index'}.issubset(filtered_df.columns):
        engagement_by_sub = filtered_df.groupby('subreddit').agg({
            'score': 'mean',
            'num_comments': 'mean',
            'upvote_ratio': 'mean',
            'engagement_index': 'mean'
        }).reset_index()

        engagement_by_sub = engagement_by_sub.sort_values('engagement_index', ascending=False).head(10)

        # Capitalize subreddit names
        engagement_by_sub['subreddit'] = engagement_by_sub['subreddit'].str.title()

        # Pie chart for engagement index
        fig = px.pie(
            engagement_by_sub,
            names='subreddit',
            values='engagement_index',
            title="🔥 Top 10 Subreddits by Engagement",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough data for engagement analysis.")

# Topic Analysis
with tab3:
    
    st.subheader("🔎 Top Topics of Discussion")

    if "topic_name" in filtered_df.columns and not filtered_df.empty:
        topic_counts = filtered_df['topic_name'].value_counts().reset_index()
        topic_counts.columns = ['topic_name', 'count']

        # Apply cleaning function to topic names
        topic_counts['topic_name'] = topic_counts['topic_name'].apply(clean_topic_name)

        # Plot the top 10 topics with labeled axes
        fig = px.bar(
            topic_counts.head(10), 
            x='topic_name', 
            y='count', 
            orientation='v', 
            title="Top 10 Topics",
            labels={'topic_name': 'Topic Name', 'count': 'Count'}  # Adding axis labels
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No topic data available.")

    # **All Topics Overview**
    st.subheader("🌎 All Topics Overview")

    if "topic_name" in filtered_df.columns:
        topic_counts = filtered_df["topic_name"].value_counts().reset_index()
        topic_counts.columns = ["topic_name", "count"]

        # Apply cleaning function to topic names
        topic_counts['topic_name'] = topic_counts['topic_name'].apply(clean_topic_name)

        if not topic_counts.empty:
            fig = px.treemap(
                topic_counts, 
                path=["topic_name"], 
                values="count", 
                title="📊 Topic Distribution",
                labels={'topic_name': 'Topic Name', 'count': 'Count'}  # Adding axis labels
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data for topic visualization.")
    else:
        st.warning("No topic data available.")

# Content Explorer 
with tab4:
    st.subheader("#️⃣ Explore Post")
        
    # Sort by options
    sort_by = st.selectbox("Sort by",
        options=["Most Recent", "Highest Score", "Most Comments", "Most Positive", "Most Negative"]
    )

    # Handle NaNs before sorting
    filtered_df['sentiment_score'] = filtered_df['sentiment_score'].fillna(0)  # Ensures NaNs do not affect sorting
    
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
        title = row['title'] if pd.notna(row['title']) else "No title available"
        selftext = row['selftext'] if pd.notna(row['selftext']) else "No content available"

        with st.expander(f"{title}"):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Title:** {title}")
                st.markdown(f"**Content:** {selftext[:500]}..." if len(selftext) > 500 else f"**Content:** {selftext}")

            with col2:
                st.markdown(f"**Subreddit:** r/{row['subreddit']}")
                st.markdown(f"**Date:** {row['created_utc'].strftime('%Y-%m-%d')}")
                st.markdown(f"**Score:** {row['score']}")
                st.markdown(f"**Comments:** {row['num_comments']}")

                # Handle NaN sentiment values
                if pd.notna(row['sentiment_score']):
                    sentiment_color = "green" if row['sentiment_score'] > 0.05 else "red" if row['sentiment_score'] < -0.05 else "gray"
                    st.markdown(f"**Sentiment:** <span style='color:{sentiment_color}'>{row['sentiment_score']:.2f}</span>", unsafe_allow_html=True)
                else:
                    st.markdown("**Sentiment:** No sentiment data available")

# Footer
st.markdown("---")
st.markdown("*Dashboard created for data visualization assignment*")

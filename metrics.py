import streamlit as st
import pandas as pd
import json
import os
import pytz
from datetime import datetime
import altair as alt

DATA_PATH = "logs/feedback_log.jsonl"

def load_feedback_data():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(columns=["timestamp", "query", "rating", "response_time"])
    with open(DATA_PATH, "r") as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]
    return pd.DataFrame(lines)

def render_dashboard():
    st.set_page_config(layout="wide")
    st.markdown("<h1>AllyIn Compass - Observability Dashboard</h1>", unsafe_allow_html=True)

    df = load_feedback_data()
    
    if df.empty:
        st.warning("No feedback data available yet.")
        return

    # Convert timestamp to PST-aware datetime
    df['date'] = df['timestamp'].str.split("T", expand=True)[0]
    df['date'] = pd.to_datetime(df['date'])

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Queries per Day")
        queries_per_day = df.groupby("date").size().reset_index(name="count")
        chart_qpd = alt.Chart(queries_per_day).mark_line(point=True).encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(format="%Y-%m-%d")),
            y=alt.Y("count:Q", title="Number of Queries"),
            color=alt.value("#26c6da"),
            tooltip=["date", "count"]
        ).properties(height=350)
        st.altair_chart(chart_qpd, use_container_width=True)

    with col2:
        st.subheader("Feedback Overview")
        feedback_counts = df["rating"].value_counts().sort_index()
        chart_df = pd.DataFrame({
            "Rating": ["Not Helpful" if i == 0 else "Helpful" for i in feedback_counts.index],
            "Count": feedback_counts.values,
            "Color": ["#ef5350" if i == 0 else "#66bb6a" for i in feedback_counts.index]
        })
        chart_feedback = alt.Chart(chart_df).mark_bar(cornerRadius=5).encode(
            x=alt.X("Rating:N", title="Feedback"),
            y=alt.Y("Count:Q", title="Count"),
            color=alt.Color("Color:N", scale=None),
            tooltip=["Rating", "Count"]
        ).properties(height=350)
        st.altair_chart(chart_feedback, use_container_width=True)

    st.subheader("Average Response Time")
    if "response_time" in df.columns and df["response_time"].notnull().any():
        avg_time = df["response_time"].dropna().mean()
        st.markdown(f"<h2 style='color:#111;'>{avg_time:.2f} sec</h2>", unsafe_allow_html=True)
    else:
        st.info("No response time data available.")

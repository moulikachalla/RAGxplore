import streamlit as st
import json
import sys
import os

# Make sure the tools module is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.rag_tool import create_rag_tool

st.set_page_config(page_title="RAGxplore", layout="wide")

# Initialize RAG tool
@st.cache_resource
def get_rag_tool():
    config = {
        "top_k": 3,
        "temperature": 0.1,
        "citation_limit": 3,
        "score_threshold": 0.3
    }
    return create_rag_tool(config)

rag_tool = get_rag_tool()

# --- Session state ---
if "rag_result" not in st.session_state:
    st.session_state.rag_result = None

if "submitted" not in st.session_state:
    st.session_state.submitted = False

# --- Sidebar ---
st.sidebar.title("🔍 Filters")
selected_domain = st.sidebar.selectbox("Select Domain", ["All", "Finance", "Biotech", "Energy"])
min_confidence = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.3, 0.05)
filter_date = st.sidebar.date_input("Filter by Date")  # Date filter
source_filter = st.sidebar.text_input("Filter by Source Filename")  

# Reset button
if st.sidebar.button("Reset"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()


# Dashboard redirect
if st.sidebar.button("Dashboard"):
    from dashboards.metrics import render_dashboard
    render_dashboard()
    st.stop()

# --- Main UI ---
st.title("🤖 RAGxplore")
st.markdown("Ask a question and let the AI assistant retrieve and summarize documents intelligently.")

user_query = st.text_input("Enter your enterprise question:")
submit = st.button("Submit Query")

# --- Handle Query Submission ---
if submit and user_query.strip():
    with st.spinner("Retrieving answer..."):
        result = rag_tool.query(user_query)
        st.session_state.rag_result = result
        st.session_state.submitted = True

# --- Use result from session state ---
result = st.session_state.rag_result

# --- Optional client-side citation filtering by filename ---
if result and source_filter.strip():
    result["citations"] = [c for c in result["citations"] if source_filter.lower() in c["source"].lower()]

if result and st.session_state.submitted:
    st.divider()

    if result["confidence"] < min_confidence:
        st.warning(f"No confident answer found above {min_confidence} threshold.")
    else:
        st.subheader("Answer")
        st.write(result["answer"])

        st.metric("Confidence Score", result["confidence"],
                  help="Weighted average of top document similarity scores from Qdrant.")
        st.markdown(f"**Sources:** {', '.join(result['sources']) or 'None'}")

        with st.expander("Citations"):
            for c in result["citations"]:
                st.markdown(f"**{c['source']}** (Score: {c['relevance_score']})")
                st.caption(c['content_preview'])

        with st.expander("Metadata"):
            st.json({
                "Query": result["question"],
                "Timestamp": result["timestamp"],
                "Model": result["model_used"],
                "Num Sources": result["num_sources"],
                "Response Time (s)": result["response_time"]
            })

        # --- Security & Compliance Section ---
        with st.expander("Security & Compliance Flags"):
            if result["pii"] or result["compliance_flags"]:
                if result["pii"]:
                    st.markdown("⚠️ **PII Detected**")
                    for pii_item in result["pii"]:
                        st.markdown(f"- **{pii_item['source']}**: {pii_item['details']}")
                if result["compliance_flags"]:
                    st.markdown("⚠️ **Compliance Risk Terms Detected**")
                    for risk in result["compliance_flags"]:
                        st.markdown(f"- **{risk['source']}**: {risk['details']}")
            else:
                st.success("✅ No PII or compliance risks found.")

        st.divider()

        # --- Feedback Section ---
        st.subheader("Was this answer helpful?")
        col1, col2 = st.columns([1, 1])
        feedback_submitted = False
        feedback = {}

        with col1:
            if st.button("👍 Yes"):
                feedback = {"query": result["question"], "answer": result["answer"], "rating": 1}
                feedback_submitted = True

        with col2:
            if st.button("👎 No"):
                feedback = {"query": result["question"], "answer": result["answer"], "rating": 0}
                feedback_submitted = True

        if feedback_submitted:
            feedback["timestamp"] = result.get("timestamp", "")
            feedback["response_time"] = result.get("response_time", 0)
            feedback_log_path = os.path.join("logs", "feedback_log.jsonl")
            os.makedirs("logs", exist_ok=True)
            with open(feedback_log_path, "a") as f:
                f.write(json.dumps(feedback) + "\n")
            st.success("✅ Feedback submitted!")

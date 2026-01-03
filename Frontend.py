import streamlit as st
import requests
import uuid

#config
API_URL = "http://localhost:8000/Chat Discussion"

st.set_page_config(
    page_title="Agentic AI",
    page_icon="🤖",
    layout="wide"
)
# Session State

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# UI Header

st.title("🤖 Agentic AI Assistant")
st.caption("Trust-aware • Evidence-backed • Agentic RAG")


# Sidebar (Trust & Metadata)

with st.sidebar:
    st.subheader("🛡️ Session Info")
    st.text(f"Session ID:\n{st.session_state.session_id}")

    st.markdown("---")
    st.subheader("ℹ️ About")
    st.write(
        """
        This assistant uses:
        - Hybrid RAG (BM25 + Vector)
        - Agentic decision making
        - Web fallback (Tavily)
        - Trust & governance metrics
        """
    )


# Chat History Display

for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.write(chat["answer"])

        # Trust block
        trust = chat["trust"]
        st.markdown(
            f"""
            **Trust Level:** `{trust['confidence']}`  
            **Evidence Count:** `{trust['evidence_count']}`  
            **Web Fallback Used:** `{trust['web_fall_back']}`  
            **Memory Used:** `{trust['memory_used']}`
            """
        )

        # Evidence block
        if chat["evidence"]:
            with st.expander("📚 Evidence"):
                for idx, ev in enumerate(chat["evidence"], 1):
                    st.markdown(
                        f"""
                        **{idx}. Source Type:** {ev['Type']}  
                        **Document:** {ev.get('document_name', 'N/A')}  
                        **Page:** {ev.get('page_number', 'N/A')}  
                        **Heading:** {ev.get('heading', 'N/A')}  
                        **Snippet:** {ev.get('snippet', 'N/A')}
                        """
                    )


# User Input

question = st.chat_input("Ask your question...")

if question:
    with st.chat_message("user"):
        st.write(question)

    payload = {
        "question": question,
        "session_id": st.session_state.session_id,
        "trace_id": True
    }

    with st.spinner("Thinking..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

        except Exception as e:
            data = {
                "answer": f"Frontend error: {str(e)}",
                "evidence": [],
                "trust": {
                    "confidence": "Low",
                    "evidence_count": 0,
                    "web_fall_back": False,
                    "memory_used": False
                }
            }

    with st.chat_message("assistant"):
        st.write(data["answer"])

        trust = data["trust"]
        st.markdown(
            f"""
            **Trust Level:** `{trust['confidence']}`  
            **Evidence Count:** `{trust['evidence_count']}`  
            **Web Fallback Used:** `{trust['web_fall_back']}`  
            **Memory Used:** `{trust['memory_used']}`
            """
        )

        if data["evidence"]:
            with st.expander("📚 Evidence"):
                for idx, ev in enumerate(data["evidence"], 1):
                    st.markdown(
                        f"""
                        **{idx}. Source Type:** {ev['Type']}  
                        **Document:** {ev.get('document_name', 'N/A')}  
                        **Page:** {ev.get('page_number', 'N/A')}  
                        **Heading:** {ev.get('heading', 'N/A')}  
                        **Snippet:** {ev.get('snippet', 'N/A')}
                        """
                    )

    # Save chat history
    st.session_state.chat_history.append({
        "question": question,
        "answer": data["answer"],
        "evidence": data["evidence"],
        "trust": data["trust"]
    })

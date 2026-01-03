This project implements a Corrective Retrieval-Augmented Generation (Corrective RAG) pipeline using LangChain and LangGraph, following an agentic AI architecture.

The workflow operates as follows:
1. The user submits a query.
2. A retriever fetches relevant document chunks.
3. A grading agent evaluates whether the retrieved content is sufficiently relevant to answer the query.
4. If relevant, the LLM generates a grounded natural language answer from the retrieved context.
5. If not relevant, the system rewrites the query, invokes a web-based search tool, and generates the final answer using external information.

This dynamic control flow — involving decision-making, tool usage, and query rewriting — reflects an advanced agentic AI pattern rather than a static pipeline.

Both backend (FastAPI) and frontend (Streamlit) outputs are available in the `Output_screenshots` directory.

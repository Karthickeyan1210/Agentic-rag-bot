import os,sys
from pathlib import Path 
Base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(Base_dir))
import numpy as np
from Bots.Data_preparation import attach_images_to_paragraphs,extract_text,detect_images,pdf_paths
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.documents import Document as LCDocument
from langchain_classic.retrievers import ContextualCompressionRetriever,EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor,LLMChainFilter,DocumentCompressorPipeline
from langchain_community.document_transformers import LongContextReorder
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq.chat_models import ChatGroq
from typing import Sequence,Annotated,List,Optional,Dict,Literal
from langchain_classic import hub
from langgraph.graph import MessagesState,StateGraph,END,START
from langgraph.prebuilt import ToolNode,tools_condition
from pydantic import BaseModel,Field
from typing_extensions import TypedDict
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_classic.agents import Tool,create_react_agent,create_self_ask_with_search_agent
from langchain_community.tools import TavilySearchResults,TavilyAnswer
load_dotenv()
from langchain_classic.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.messages import AIMessage,HumanMessage
from langchain_classic.memory import ConversationBufferWindowMemory

ASTRA_DB_API_ENDPOINT=os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
AstraDB_Keyspace = 'default_keyspace'

GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
Tavily_API_Key = os.getenv("TAVILY_API_KEY") or "tavily_xxx"
Langchain_API_Key = os.getenv("LANGCHAIN_API_KEY") or "lsmith_xxx"
GROQ_API_KEY= os.getenv("GROQ_API_KEY") or "groq_xxx"
# Set environment variables
os.environ["TAVILY_API_KEY"] = Tavily_API_Key
os.environ["LANGCHAIN_API_KEY"] = Langchain_API_Key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5",encode_kwargs={"normalize_embeddings": True})
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    api_key=GROQ_API_KEY,
    timeout=None,
    max_retries=2,
)

def sanitize_metadata(metadata):
    clean_meta = {}

    for k, v in metadata.items():
        key = k.replace(" ", "_").lower()   
        if isinstance(v, Path):
            clean_meta[key] = str(v)
        elif isinstance(v, list):
            clean_meta[key] = [
                str(i) if isinstance(i, Path) else i
                for i in v
            ]
        elif isinstance(v, (str, int, float, bool)) or v is None:
            clean_meta[key] = v
        else:
            clean_meta[key] = str(v)

    return clean_meta

# 1. Extract text
doc = extract_text(pdf_paths)
structured_data = doc["Paragraphs"]

# 2. Extract images
images = detect_images(pdf_paths)

# 3. Attach images
structured_data = attach_images_to_paragraphs(structured_data, images)


def chunk_text(structured_data,chunk_size=550,chunk_overlap=300):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,separators=["\n\n", "\n", ".", " ", ""])
    text_chunks = []

    for record in structured_data:
        metadata = {
            "Document_name":record.get("Document_name"),
            "Page num":record.get("page_num"),
            "Heading":record.get("heading"),
            "Sub Heading":record.get("sub_heading"),
            "Sub_sub_heading":record.get("sub_sub_heading"),
            "snippet":record.get("images"),
        }

        for chunk in text_splitter.split_text(record.get("paragraph")):
            text_chunks.append({"text":chunk,"metadata":metadata})
    return text_chunks

chunks = chunk_text(structured_data)
all_chunks=[]
for i, chunk in enumerate(chunks):
    # print(i, type(chunk.get("metadata")), chunk.get("metadata"))
    # if not isinstance(chunk.get("metadata"), dict):
    #     print("INVALID METADATA FOUND AT INDEX:", i)
    #     break
    clean_metadata = sanitize_metadata(chunk['metadata'])
    all_chunks.append(Document(page_content=chunk['text'],metadata=clean_metadata))

#2.Embedding
sample_query = "Definition post-market surveillance"
sample_embed = embeddings.embed_query(sample_query)
# print(f"First five dimensions of embeddings were:{sample_embed[:5]}")

#3. Vector store
vector_store = AstraDBVectorStore(
    embedding=embeddings,
    collection_name="Bot_collection_HIL",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token = ASTRA_DB_APPLICATION_TOKEN,
    namespace=AstraDB_Keyspace
)

#4. Injestion:
# vector_store.add_documents(all_chunks)
# print(f"Total chunks created were {len(all_chunks)}")
# print(f" All documents ingested successfully {len(all_chunks)}")

# #5. Retriever
reorder = LongContextReorder()
FlashrankRerank.model_rebuild()
reranker = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=10)
# compressor = LLMChainExtractor.from_llm(llm)
# filter = LLMChainFilter.from_llm(llm)
keyword_retriever = BM25Retriever.from_documents(all_chunks)
keyword_retriever.k=5
vector_retriever = vector_store.as_retriever(search_type="similarity",kwargs={"k":5})
hybrid_retriever= EnsembleRetriever(retrievers=[keyword_retriever,vector_retriever],weights=[0.5,0.5])
pipeline = DocumentCompressorPipeline(transformers=[reranker])
retriever = ContextualCompressionRetriever(base_compressor=pipeline,base_retriever=hybrid_retriever)

# if __name__=="__main__":
#     query = "What is Negation?"
#     answer =retriever.invoke(query)
#     print(answer)

#Memory:
memory = ConversationBufferWindowMemory(k=17, memory_key="Chat_History",return_messages=True)
def format_chat_history():
    """Read memory and format as string for prompt"""
    chat_history_str = ""
    for message in memory.chat_memory.messages:
        if isinstance(message,HumanMessage):
            chat_history_str +=f"Human message:{message.content}\n"
        elif isinstance(message,AIMessage):
            chat_history_str +=f"Human message:{message.content}\n"
    return chat_history_str.strip()

#Corrective RAG--->Agentic AI

def make_msgpack_safe(obj):
    if isinstance(obj, dict):
        return {k: make_msgpack_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_msgpack_safe(v) for v in obj]
    elif isinstance(obj, np.generic):  # numpy.float32, int64, etc.
        return obj.item()
    elif isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)

#Web Prompt:
web_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert assistant.
Use web search results to answer the user's question.
Do NOT say 'I don't know'.
You can also respond to general greetings like "hi", "hello", "who are you", etc. Ignore spelling mistakes.
If spelling mistakes from user understand the intent what they were trying to ask. If people asking some tricky psycological question about your confidence, handle diplomatically.
     
     """),
    ("human", """Conversation so far:
{chat_history}

Current Question:
{question}

Web Results:
{context}
""")
])

#1. Grader node
class grade_docs(BaseModel):
    binary_score: str=Field("Grade the retrieved documents with its relevance and say yes or no")
llm_with_structured_output = llm.with_structured_output(grade_docs)

system = """You are a relevance grader for Corrective RAG.

Rules:
- Compare the document and the user question.
- If the document is relevant, output "Yes".
- If the document is irrelevant, empty, or off-topic, output "No".
- Output ONLY "Yes" or "No". """

prompt = ChatPromptTemplate([
    ("system", system),
    ("human", """Conversation so far:
{chat_history}

Retrieved Document:
{documents}

User Question:
{question}
""")
])
grader = prompt | llm_with_structured_output
# question = "question"
# docs = retriever.invoke(question)
# if not docs or len(docs)==0:
#     result={
#         "binary_score":"No",
#         "reason": "No relevant documents found"
#     }
#     # print(result)
# else:
#     context = "\n".join(d.page_content for d in docs[:3] if d.page_content)

#     if not context.strip():
#         result = { "binary_score":"No",
#         "reason": "Documents are not that releavant"}

#     else:
#         grade_answer = grader.invoke({"question":question,"documents":docs[1].page_content})

    # print(grade_answer)

#2. Generate node
rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant for question-answering tasks.

Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Conversation history is provided for context only.
Do NOT use conversation history as a source of truth unless supported by retrieved context.

You can also respond to general greetings like "hi", "hello", "who are you", etc. Ignore spelling mistakes.
If spelling mistakes from user understand the intent what they were trying to ask. If people asking some tricky psycological question about your confidence, handle diplomatically.

"""
        ),
        (
            "human",
            """Conversation History:
{chat_history}

Retrieved Context:
{context}

Question:
{question}
"""
        )
    ]
)

output_parser = StrOutputParser()


def format_documents(documents):
    formatted = []

    for i, d in enumerate(documents[:3], 1):
        meta = d.get("metadata", {})

        formatted.append(
            f"""[Document {i}]
Source: {meta.get("document_name", "Unknown")}
Page: {meta.get("page_num", "N/A")}
Heading: {meta.get("heading", "N/A")}
Sub Heading: {meta.get("sub_heading", "N/A")}
Sub-sub Heading: {meta.get("sub_sub_heading", "N/A")}
Images: {meta.get("snippet", "N/A")}

Content:
{d["page_content"]}
"""
        )

    return "\n\n".join(formatted)



rag_chain = (
    {
        "context": lambda x: format_documents(x['documents']),
        "question": lambda x : x['question'],
        "chat_history": lambda _: format_chat_history(), 
        
    }
    |rag_prompt
    |llm
    |output_parser
)

web_chain = (
    {
        "context": lambda x: format_documents(x['documents']),
        "question": lambda x: x['question'],
        "chat_history": lambda _: format_chat_history(),
    }
    |web_prompt
    |llm
    |output_parser
)
# generate = rag_chain.invoke({"question":"what is negation","documents":docs})


#3.Rewrite Query
system = """You are a question rewriter, rewrite the question from the user in an optimized way without changin its semantic meaning for web search"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","Here is the initial user question : {question} formualte in a better optimized way ")
    ]
)
rewriter = prompt|llm|output_parser
# print(rewriter.invoke("What is the name shake value"))

#4.Websearch tool
google_search = TavilySearchResults()
def tavily_tool(question):
    result = google_search.invoke({"query":question})
    return result

#Create Pydantic
class DocMetadata(BaseModel):
    document_name: str
    page_num: int
    heading: Optional[str]
    sub_heading: Optional[str]
    sub_sub_heading: Optional[str]
    snippet: Optional[List[str]] = []

class SerializableDoc(BaseModel):
    page_content: str
    metadata: dict
    

class Agent_type(TypedDict):
    question: str
    generation: str | None
    web_search: str
    documents: List[dict]

def serialize_docs(docs: list[LCDocument]) -> list[SerializableDoc]:
    return [
        {
            "page_content": d.page_content,
            "metadata": make_msgpack_safe(d.metadata)
        }
        for d in docs
    ]

#1.Retriever node
def retriever_node(state: Agent_type):
    print("____Retriever____")

    question = state["question"]
    history = format_chat_history()

    enriched_query = f"""
Conversation Context:
{history}

Current Question:
{question}
""".strip()

    try:
        docs = retriever.invoke(enriched_query)
        documents = serialize_docs(docs)
    except Exception as e:
        print("Retriever failed:", e)
        documents = []

    return {
        "question": question,
        "documents": documents,
        "web_search": "no",
        "generation": None
    }

# retrieved_state = retriever_node({"question":"What is negation"})
# print(retrieved_state)

#2. Generate node
def generate_node(state: Agent_type):
    print("____Generate____")

    question = state["question"]
    documents = state["documents"]

    if state["web_search"] == "yes":
        generation = web_chain.invoke(state)
    else:
        generation = rag_chain.invoke(state)

    #Save conversation
    memory.save_context(
        {"input": question},
        {"output": generation}
    )

    return {
        "question": question,
        "documents": documents,
        "generation": generation,
        "web_search": state["web_search"]
    }

#3. Grade_node
def grade_node(state: Agent_type):
    print("____Grader____")

    question = state["question"]
    documents = state.get("documents", [])

    filtered_docs = []
    web_search = "no"

    for doc in documents:
        content = doc["page_content"]

        if not content.strip():
            web_search = "yes"
            continue

        score = grader.invoke({
            "question": question,
            "documents": content,
            "chat_history": format_chat_history()
        })

        if score.binary_score == "Yes":
            filtered_docs.append(doc)
        else:
            web_search = "yes"

    if not filtered_docs:
        web_search = "yes"

    return {
        "question": question,
        "documents": filtered_docs,
        "web_search": web_search
    }


#4. Rewriter Node
def rewriter_node(state:Agent_type):
    question = state['question']
    rewrited_query = rewriter.invoke({"question":question})
    return {"question":rewrited_query,"documents": state["documents"],
        "web_search": state["web_search"]}

#5. Web search
def web_search_node(state: Agent_type):
    question = state["question"]
    documents = state["documents"]

    results = tavily_tool(question)
    web_text = "\n".join(
        r["content"] if isinstance(r, dict) else str(r)
        for r in results
    )

    documents.append({
        "page_content": web_text,
        "metadata": {"source": "web"}
    })

    return {
        "question": question,
        "documents": documents
    }


#6. Decision Maker
def decision_to_generate_node(state:Agent_type):
    question = state['question']
    documents = state['documents']
    web_search = state['web_search']

    if web_search == "yes":
        print("Question will be transformed by rewriting the query in a way that web search can able to understand")
        return "rewriter_node"
    
    else:
        return "generate_node"

workflow = StateGraph(Agent_type)
workflow.add_node("Retriever",retriever_node)
workflow.add_node("Generation",generate_node)
workflow.add_node("Web search",web_search_node)
workflow.add_node("Rewriter",rewriter_node)
workflow.add_node("Grader",grade_node)
workflow.set_entry_point("Retriever")
workflow.add_edge("Retriever","Grader")
workflow.add_conditional_edges("Grader",decision_to_generate_node,{"rewriter_node":"Rewriter","generate_node":"Generation"})
workflow.add_edge("Rewriter","Web search")
workflow.add_edge("Web search","Generation")
workflow.add_edge("Generation",END)
app = workflow.compile(checkpointer=MemorySaver())
config = {"configurable":{"thread_id":1}}


def Agent_Result(question: str) -> str:
    answer = app.invoke({"question": question}, config, stream_mode="values")
    gen = answer.get('generation')
    if not gen or gen is None:
        return "No relevant answer found."
    return str(gen)

print(Agent_Result("what is use of tavily search"))
    

# print(result("What is negation"))
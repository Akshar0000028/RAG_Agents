from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from operator import add as add_message
import os

class AdaptiveRagAgent(TypedDict):
    query: Annotated[Sequence[BaseMessage], add_message]
    retry_count: int
    answer: str

llm = ChatNVIDIA(
    model="meta/llama3-70b-instruct",
    nvidia_api_key="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",
    temperature=0.3
)

embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    model_type="passage",
    nvidia_api_key="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"
)

pdf_path = "C:\\Users\\Akshar Savaliya\\Downloads\\Agentic_NN.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"File not found {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)
pages = pdf_loader.load()
print(f"PDF loaded successfully with {len(pages)} pages")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = text_splitter.split_documents(pages)
vectorstore = Chroma.from_documents(documents, embeddings)

def query_analysis_node(state):
    query = state["query"][0].content.lower()
    if any(term in query for term in ["latest", "2025", "current", "recent"]):
        return {"path": "web_search"}
    elif len(query) < 20 or "summary" in query:
        return {"path": "direct"}
    return {"path": "rag"}

def web_search_node(state):
    query = state["query"][-1].content
    return {"web_search_result": f"Web search results for: {query} (mock data)"}

def retriever_node(state):
    query = state["query"][-1].content
    docs = vectorstore.similarity_search(query, k=4)
    return {"retrieved_docs": docs}

def grade_node(state):
    retrieved_docs = state.get("retrieved_docs", [])
    query = state["query"][-1].content.lower()
    query_terms = query.split()
    relevant = any(
        any(term in doc.page_content.lower() for term in query_terms) or
        any(term in doc.page_content.lower() for term in ["agentic", "neural", "network"])
        for doc in retrieved_docs
    )
    return {"relevant": relevant}

def generate_node(state):
    web_context = state.get("web_search_result", "")
    retrieved_texts = "\n".join(doc.page_content for doc in state.get("retrieved_docs", []))
    context = f"Web Context:\n{web_context}\n\nDocument Context:\n{retrieved_texts}"
    prompt = f"{context}\n\nQuestion: {state['query'][-1].content}"

    try:
        response = llm.invoke(prompt)
        return {
            "query": [state["query"][-1]],
            "retry_count": state.get("retry_count", 0),
            "answer": response.content if hasattr(response, "content") else str(response)
        }
    except Exception as e:
        return {
            "query": [state["query"][-1]],
            "retry_count": state.get("retry_count", 0),
            "answer": f"Error generating response: {e}"
        }

def rewrite_node(state):
    count = state.get("retry_count", 0)
    if count >= 2:
        return {
            "query": [state["query"][-1]],
            "retry_count": count,
            "force_generate": True  
        }
    
    original_query = state["query"][-1].content
    new_query_text = f"Explain in detail about {original_query}"
    return {
        "query": [HumanMessage(content=new_query_text)],
        "retry_count": count + 1
    }

def self_reflect_node(state):
    answer = state.get("answer", "").lower()
    hallucinated = "error" in answer or "don't know" in answer or "not found" in answer
    return {"hallucinated": hallucinated}


graph = StateGraph(AdaptiveRagAgent)
graph.set_entry_point("query_analysis")

graph.add_node("query_analysis", query_analysis_node)
graph.add_node("web_search", web_search_node)
graph.add_node("retriever", retriever_node)
graph.add_node("grade", grade_node)
graph.add_node("rewrite", rewrite_node)
graph.add_node("generate", generate_node)
graph.add_node("self_reflect", self_reflect_node)

graph.add_conditional_edges(
    "query_analysis",
    lambda s: s["path"],
    {"direct": "generate", "rag": "retriever", "web_search": "web_search"}
)

graph.add_edge("web_search", "generate")
graph.add_edge("generate", "self_reflect")

graph.add_conditional_edges(
    "self_reflect",
    lambda s: "generate" if s["hallucinated"] else END,
    {"generate": "generate", END: END}
)

graph.add_edge("retriever", "grade")

graph.add_conditional_edges(
    "grade",
    lambda s: "generate" if s["relevant"] else "rewrite",
    {"generate": "generate", "rewrite": "rewrite"}
)

graph.add_conditional_edges(
    "rewrite",
    lambda s: "generate" if s.get("force_generate", False) else "retriever",
    {"generate": "generate", "retriever": "retriever"}
)


app = graph.compile()

query = HumanMessage(content="What is Structure of the Agentic Neural Network?")
result = app.invoke({
    "query": [query],
    "retry_count": 0,
    "answer": ""
})

print("Final Answer:")
print(result.get("answer", "No answer was generated"))
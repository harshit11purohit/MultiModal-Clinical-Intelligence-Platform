from flask import Flask, render_template, request, jsonify  # Restored render_template
from dotenv import load_dotenv
import os

from pinecone import Pinecone
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from langchain_pinecone import PineconeVectorStore

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever

from langchain_classic.chains import create_history_aware_retriever
from langchain_groq import ChatGroq

from src.utils import (
    download_hugging_face_embeddings,
    load_pdf
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = "langchain-medical-chatbot"

app = Flask(__name__)

SESSION_MEMORY = {}

# Load Base Embedding models
embeddings = download_hugging_face_embeddings()
pc = Pinecone(api_key=PINECONE_API_KEY)

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# Base Retrieval Setup
dense_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 15}
)

documents = load_pdf("data/")
tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    return len(tokenizer.encode(text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=tiktoken_len
)

bm25_docs = []
for page in documents:
    chunks = splitter.split_text(page.page_content)
    for chunk in chunks:
        bm25_docs.append(Document(page_content=chunk, metadata=page.metadata))

bm25_retriever = BM25Retriever.from_documents(bm25_docs)
bm25_retriever.k = 15

ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)

cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=4)

base_compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=ensemble_retriever
)

# Core Clinical Processing Engine
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-20b",
    temperature=0.1,
    max_tokens=512
)

# Intent Router Setup
router_llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-20b",
    temperature=0.0
)

router_prompt_template = """You are an advanced medical query traffic controller. 
Analyze the current user input along with the past chat history. You must determine if the query requires looking up private document data, or if it is a general medical definition/greeting.

Classify it into exactly one of these two categories:
- GENERAL: If the user is greeting you, asking broad medical questions ("What is a vitamin?"), or asking conversational questions that do not reference private clinic data or specific custom records.
- RAG: If the user is explicitly asking about their specific medical records, uploaded document content, customized health plan details, or continuing a tracking thread about a private document entry.

Respond with exactly ONE word: either GENERAL or RAG. Do not include any punctuation or extra text.

Chat History:
{chat_history}

Current Input: {input}
Output:"""

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, base_compression_retriever, contextualize_q_prompt
)

# Production Prompts
rag_prompt_template = """
You are a compassionate, world-class AI Clinical Physician. Your tone is professional, clear, and empathetic.
Use the provided Context as your primary foundational source of truth.

Context:
{context}

Question:
{question}
Answer:"""
RAG_PROMPT = PromptTemplate(template=rag_prompt_template, input_variables=["context", "question"])

general_prompt_template = """
You are a compassionate, world-class AI Clinical Physician. Answer comprehensively using your global medical knowledge bank.

Question:
{question}
Answer:"""
GENERAL_PROMPT = PromptTemplate(template=general_prompt_template, input_variables=["question"])


def answer_question_with_advanced_routing(session_id, question):
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []
        
    chat_history = SESSION_MEMORY[session_id]
    
    history_str = "\n".join([f"{type(m).__name__}: {m.content}" for m in chat_history[-4:]])
    router_prompt = router_prompt_template.format(chat_history=history_str, input=question)
    intent = router_llm.invoke(router_prompt).content.strip().upper()
    
    citations = []
    
    if "RAG" in intent:
        print("[Traffic Router] -> Selected RAG Pipeline path.")
        docs = history_aware_retriever.invoke({"input": question, "chat_history": chat_history})
        context = "\n\n".join(doc.page_content for doc in docs)
        
        for doc in docs:
            source_info = {
                "file": os.path.basename(doc.metadata.get("source", "Unknown_Document")),
                "page": doc.metadata.get("page", "N/A")
            }
            if source_info not in citations:
                citations.append(source_info)
                
        final_prompt = RAG_PROMPT.format(context=context, question=question)
    else:
        print("[Traffic Router] -> Selected General Knowledge Base path.")
        final_prompt = GENERAL_PROMPT.format(question=question)
        
    response = llm.invoke(final_prompt)
    answer = response.content
    
    SESSION_MEMORY[session_id].append(HumanMessage(content=question))
    SESSION_MEMORY[session_id].append(AIMessage(content=answer))
    
    if len(SESSION_MEMORY[session_id]) > 10:
        SESSION_MEMORY[session_id] = SESSION_MEMORY[session_id][-10:]
        
    return {"answer": answer, "citations": citations}

# --- FIXED: Restored home route to serve your local chat.html frontend ---
@app.route("/", methods=["GET"])
def index():
    return render_template("chat.html")

# --- FIXED: Handles standard HTML form data mapping cleanly ---
@app.route("/get", methods=["POST"])
def chat():
    if request.is_json:
        data = request.get_json()
        user_input = data.get("msg", "")
        session_id = data.get("session_id", "default_user")
    else:
        user_input = request.form.get("msg", "")
        session_id = request.form.get("session_id", "default_user")
    
    try:
        result = answer_question_with_advanced_routing(session_id, user_input)
        return jsonify(result), 200
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"answer": "An internal RAG error occurred.", "citations": []}), 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)
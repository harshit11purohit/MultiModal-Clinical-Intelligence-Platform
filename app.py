from flask import Flask, render_template, request, jsonify
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
# Modern sub-chains for memory management
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

# Memory Store to manage separate chat histories per user session
# Structure: { "session_id": [HumanMessage(...), AIMessage(...)] }
SESSION_MEMORY = {}

# Load Base Embeddings
embeddings = download_hugging_face_embeddings()
pc = Pinecone(api_key=PINECONE_API_KEY)

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# 1. Base Retrieval Candidate Infrastructure
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

# Combine Dense + Sparse Hybrid Inputs
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)

# Deep Reranking Engine Construction
cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=4)

base_compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=ensemble_retriever
)

# 2. Modern LLM Config
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-20b",
    temperature=0.1,
    max_tokens=512
)

# 3. --- HISTORY AWARE REPHRASING SUB-CHAIN ---
# This instruction forces the model to look at past texts and clean up ambiguous references
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Wrap your entire two-stage retriever with history rephrasing intelligence!
history_aware_retriever = create_history_aware_retriever(
    llm, base_compression_retriever, contextualize_q_prompt
)

# Core Prompt Layout for final output delivery
prompt_template = """
You are a compassionate, world-class AI Clinical Physician and Medical Assistant. Your tone is professional, clear, and empathetic.

Handling Greetings:
- If the user's question is a simple greeting or conversational pleasantry (e.g., "hi", "hello", "hey", "good morning"), bypass database processing entirely and reply with a warm, welcoming clinical greeting (e.g., "Hello! I am your AI medical assistant. How can I help you with your health questions today?").

Core Clinical Directives (For all health queries):
- Use the provided Context as your primary foundational source of truth. 
- You are explicitly permitted to use your own advanced medical training to interpret, synthesize, explain, and smoothly connect the concepts found in the context so the response flows naturally.
- Do not blindly copy-paste snippets. Translate raw medical data into patient-friendly, professional, and clear language.
- If the core information required to answer the question is entirely missing from both the context and your general knowledge base, state honestly: "I don't know based on the provided medical documents, and I cannot find a safe medical baseline to answer."
- Keep your answers highly structured: use a brief introductory paragraph followed by clean, actionable bullet points for symptoms, instructions, or risks whenever appropriate.

-------------------------
Context:
{context}
-------------------------
Question:
{question}
-------------------------
Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

def answer_question_with_memory(session_id, question):
    # Ensure memory array exists for this user profile session
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []
    
    chat_history = SESSION_MEMORY[session_id]
    
    # Pass the question + chat history to extract the context docs via rephrasing sub-chain
    docs = history_aware_retriever.invoke({"input": question, "chat_history": chat_history})
    
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Format finalized template instructions
    final_prompt = PROMPT.format(context=context, question=question)
    
    response = llm.invoke(final_prompt)
    answer = response.content
    
    # Append conversation turns into the tracking history
    SESSION_MEMORY[session_id].append(HumanMessage(content=question))
    SESSION_MEMORY[session_id].append(AIMessage(content=answer))
    
    # Limit memory context log scope to top 10 messages so token footprint stays light
    if len(SESSION_MEMORY[session_id]) > 10:
        SESSION_MEMORY[session_id] = SESSION_MEMORY[session_id][-10:]
        
    return answer

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST", "GET"])
def chat():
    user_input = request.form["msg"]
    
    # Use a fallback identifier for simple single-user web frameworks.
    # If your front-end passes a distinct session identifier via requests, extract it here.
    session_id = request.form.get("session_id", "default_user")
    
    print(f"\nUser : {user_input}")
    
    try:
        answer = answer_question_with_memory(session_id, user_input)
        print(f"\nAssistant : {answer}")
        return answer
    except Exception as e:
        print(f"Execution Error: {e}")
        return "Sorry, an internal error occurred while generating the response."
    
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True,
        use_reloader=False
    )
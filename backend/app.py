import os
import tempfile
import pickle
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

# LangChain Core & Loaders
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    PyPDFLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate

# Advanced Retrieval & Reranking
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from PIL import Image

from src.utils import download_hugging_face_embeddings

from langchain_groq import ChatGroq

load_dotenv()

# --- LangSmith Integration ---
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


BM25_CACHE_FILE = "bm25_docs_cache.pkl"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = "langchain-medical-chatbot"

# Deduplication set
processed_files = set()

if not os.environ.get("USER_AGENT"):
    os.environ["USER_AGENT"] = "MedicalChatbot/1.0"

app = Flask(__name__)
CORS(
    app,
    origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost",  # Adds your Docker Nginx port
        "http://127.0.0.1",  # Adds your Docker Nginx IP
    ],
)

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

fast_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
print("🤖 Initializing Groq LLM (llama-3.1-8b-instant)...")
llm = None
if not GROQ_API_KEY:
    print(
        "⚠️ Warning: GROQ_API_KEY not found in .env. Add GROQ_API_KEY=your_key to .env"
    )
else:
    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=1024,
        )
        print("✅ Groq LLM initialized successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize Groq LLM. Error: {e}")
        llm = None


def llm_call(prompt: str) -> str:
    """
    Thin wrapper so the rest of the code can keep calling `llm_call(prompt)`
    """
    if llm is None:
        return ""
    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        print(f"❌ Groq call failed: {e}")
        return ""


print("🧠 Loading HuggingFace Embedding Model...")
embeddings = download_hugging_face_embeddings()

print("🌲 Connecting to Cloud Pinecone Vector Index...")
pinecone_vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
pinecone_retriever = pinecone_vectorstore.as_retriever(search_kwargs={"k": 5})

# Just load the existing cache, don't try to re-ingest the 'data/' folder
if os.path.exists(BM25_CACHE_FILE):
    with open(BM25_CACHE_FILE, "rb") as f:
        bm25_docs = pickle.load(f)
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = 5
else:
    bm25_docs = []
    # Set up an empty retriever so the app doesn't crash
    bm25_retriever = BM25Retriever.from_documents(
        [Document(page_content="No documents indexed.")]
    )
    bm25_retriever.k = 1

print("⚖️ Creating Hybrid Search Ensemble...")
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, pinecone_retriever], weights=[0.5, 0.5]
)

print("🎯 Setting up Cross-Encoder Reranker...")
cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)

# =====================================================================
# 2. Application Routes (API Endpoints)
# =====================================================================


@app.route("/", methods=["GET"])
def index():
    """Simple root route so hitting the base URL doesn't 404."""
    return jsonify(
        {
            "status": "ok",
            "service": "Medical Chatbot API",
            "endpoints": ["/health", "/chat (POST)", "/upload (POST)"],
        }
    )


@app.route("/health", methods=["GET"])
def health():
    """Health check route to confirm the server + LLM are reachable."""
    return jsonify(
        {
            "status": "ok",
            "llm_initialized": llm is not None,
            "bm25_chunks_loaded": len(bm25_docs),
        }
    )


def generate_clinical_response(user_query: str, extra_context: str = "") -> dict:
    """
    Shared pipeline: Hybrid Retrieval (BM25 + Pinecone) -> Cross-Encoder Rerank ->
    Structured Clinical Prompt -> Safety Guardrail.

    Used by BOTH /chat and /upload so the two entry points (ask-only, and the new
    unified upload+question flow) always go through identical Reranking and
    Guardrail logic, instead of two copies drifting apart.

    extra_context: freshly-uploaded document text (if any). It is placed first in
    the context window since it's the most relevant/most recent source for the
    user's question, ahead of whatever the ensemble retriever pulls from the
    long-term index.
    """
    retrieved_docs = ensemble_retriever.invoke(user_query)
    final_reranked_docs = reranker.compress_documents(retrieved_docs, user_query)

    context_string = "\n".join([d.page_content for d in final_reranked_docs])

    if extra_context:
        context_string = f"{extra_context}\n\n---\n\n{context_string}"

    # Enhanced Report Analysis Structure
    final_prompt = f"""You are an elite Clinical AI Specialist. Use this context to answer:
    {context_string}
    Query: {user_query}
    Format strictly using:
    - Executive Summary
    - Report Interpretation
    - Abnormal Parameters
    - Possible Causes
    - Dietary Advice
    - Lifestyle Changes
    - Medicines Mentioned
    - When To Consult Doctor
    - Red Flags
    - Sources"""

    complete_answer = llm_call(final_prompt).strip()

    # Safety Guardrail
    guardrail_prompt = f"""Review this AI answer. If the answer is helpful, factual, and not dangerous, reply 'PASS'. 
    If the answer is promoting self-harm, illegal acts, or is blatantly hallucinating dangerous misinformation, reply 'REJECT'.
    Answer: {complete_answer}
    Decision:"""

    validation = llm_call(guardrail_prompt).strip().upper() or "PASS"
    if "REJECT" in validation:
        return {
            "response": "For your safety, I cannot provide an unverified medical answer. Please consult a licensed physician regarding this query.",
            "guardrail_triggered": True,
        }

    return {"response": complete_answer}


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_input = (data.get("msg") or "").strip()

    if not user_input:
        return jsonify({"error": "Empty query string provided."}), 400

    # 1. Improved Strict Routing
    router_prompt = f"""Return exactly one word. 
    'REPORT' if they want report analysis, 'GENERAL' for health questions, or 'GREETING' for simple talk.
    Question: {user_input}
    Decision:"""
    route_decision = llm_call(router_prompt).strip().upper()

    if "GREETING" in route_decision:
        return jsonify(
            {"response": llm_call(f"Reply politely and briefly to: {user_input}")}
        )

    return jsonify(generate_clinical_response(user_input))


@app.route("/upload", methods=["POST"])
def upload_data():
    global bm25_docs, bm25_retriever, ensemble_retriever

    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    # NEW: Unified Upload & Query — an optional question travels in the same
    # multipart/form-data request as the file (request.form, not request.files).
    user_query = (request.form.get("msg") or "").strip()

    # 3. Deduplication Check
    if file.filename in processed_files:
        # Even if this exact file was already indexed, the user may still be
        # asking a fresh question about it (or about the corpus in general) —
        # answer it instead of just reporting "already processed".
        if user_query:
            return jsonify(generate_clinical_response(user_query))
        return jsonify({"response": "File already processed."}), 200

    processed_files.add(file.filename)

    content_for_summary = ""

    # Handle PDF Upload
    if file.filename.lower().endswith(".pdf"):
    # Create the temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        try:
        # Save and process
            file.save(tmp.name)
            tmp.close()  # <--- CRITICAL: Close the file handle before loader uses it
        
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()
        
            content_for_summary = "\n".join([doc.page_content for doc in docs])
            chunks = fast_splitter.split_documents(docs)
        
            PineconeVectorStore.from_documents(
                chunks, embeddings, index_name=INDEX_NAME
            )
            bm25_docs.extend(chunks)
        
        finally:
        # Clean up: close if still open, then delete
            if not tmp.closed:
                tmp.close()
            if os.path.exists(tmp.name):
                os.remove(tmp.name)
    else:
        # 4. Improved OCR
        image = Image.open(file).convert("L")
        extracted_text = pytesseract.image_to_string(image, config="--oem 3 --psm 6")
        content_for_summary = extracted_text

        doc = Document(page_content=extracted_text, metadata={"source": file.filename})
        chunks = fast_splitter.split_documents([doc])
        PineconeVectorStore.from_documents(chunks, embeddings, index_name=INDEX_NAME)
        bm25_docs.extend(chunks)

    # Save cache
    with open(BM25_CACHE_FILE, "wb") as f:
        pickle.dump(bm25_docs, f)

    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, pinecone_retriever], weights=[0.5, 0.5]
    )

    # NEW: Unified Upload & Query — if the user attached a question along with
    # the file, answer THAT question (using the fresh document as priority
    # context) instead of just returning a generic summary. Goes through the
    # exact same Reranking + Guardrail pipeline as /chat.
    if user_query:
        return jsonify(
            generate_clinical_response(user_query, extra_context=content_for_summary)
        )

    # Fallback (legacy behavior): no specific question was asked, so produce
    # a full automatic summary of the uploaded document.
    analysis_prompt = f"""You are an elite Clinical AI Specialist. Analyze this document:
    {content_for_summary}
    Format strictly using:
    - Executive Summary
    - Report Interpretation
    - Abnormal Parameters
    - Possible Causes
    - Dietary Advice
    - Lifestyle Changes
    - Medicines Mentioned
    - When To Consult Doctor
    - Red Flags
    - Sources"""

    auto_summary = llm_call(analysis_prompt)

    return jsonify({"response": auto_summary}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)

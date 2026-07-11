import os
import tempfile
import pickle
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

# LangChain Core & Loaders
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate

# Advanced Retrieval & Reranking
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Image Processing (OCR)
import pytesseract
from PIL import Image

# Utility imports from your local project framework
from src.utils import download_hugging_face_embeddings

# --- GROQ (replaces local CTransformers/Llama2) ---
# pip install langchain-groq
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = "langchain-medical-chatbot"

# Fix for USER_AGENT warning thrown by WebBaseLoader
if not os.environ.get("USER_AGENT"):
    os.environ["USER_AGENT"] = "MedicalChatbot/1.0"

# Initialize Headless Flask App for Lovable AI Frontend
app = Flask(__name__)
CORS(app)  # Enables React/Lovable frontend to talk to this API safely

# Increased size limit for proper, large PDF uploads (50 MB limit)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# =====================================================================
# 1. Pipeline Initialization & Global State
# =====================================================================

print("🤖 Initializing Groq LLM (openai/gpt-oss-120b)...")
llm = None
if not GROQ_API_KEY:
    print("⚠️ Warning: GROQ_API_KEY not found in .env. Add GROQ_API_KEY=your_key to .env")
else:
    try:
        # openai/gpt-oss-120b is Groq's current flagship production model
        # (not on their deprecation list, unlike llama-3.3-70b-versatile / llama-3.1-8b-instant,
        # which are being retired on 08/16/2026). Swap the model string below if you prefer
        # another currently-supported Groq model, e.g. "qwen/qwen3.6-27b".
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="openai/gpt-oss-120b",
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
    exactly like the old `llm(prompt)` CTransformers-style call, even though
    ChatGroq (a chat model) returns a message object instead of a raw string.
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

# --- ADVANCED: LOCAL BM25 CACHING FOR 1000+ PAGES ---
BM25_CACHE_FILE = "bm25_docs_cache.pkl"
fast_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

global bm25_retriever, bm25_docs, ensemble_retriever
bm25_docs = []

if os.path.exists(BM25_CACHE_FILE):
    print(f"⚡ Loading Fast BM25 Index from local cache ({BM25_CACHE_FILE})...")
    with open(BM25_CACHE_FILE, "rb") as f:
        bm25_docs = pickle.load(f)
    print(f"✅ Instantly loaded {len(bm25_docs)} chunks from disk.")
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = 5
else:
    print("📄 Ingesting local PDFs for Fast BM25 Sync (First time only)...")
    loader = PyPDFDirectoryLoader("data/")
    local_raw_docs = loader.load()

    if local_raw_docs:
        bm25_docs = fast_splitter.split_documents(local_raw_docs)
        print(f"✅ Formatted {len(bm25_docs)} local chunks. Saving cache to disk...")
        with open(BM25_CACHE_FILE, "wb") as f:
            pickle.dump(bm25_docs, f)
        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = 5
    else:
        bm25_retriever = BM25Retriever.from_documents([Document(page_content="Empty context")])
        bm25_retriever.k = 1

print("⚖️ Creating Hybrid Search Ensemble...")
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, pinecone_retriever],
    weights=[0.5, 0.5]
)

print("🎯 Setting up Cross-Encoder Reranker...")
cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)

# --- Master Structural Prompt ---
CUSTOM_RAG_PROMPT_TEMPLATE = """You are an elite Clinical AI Specialist. Provide comprehensive, accurate, and empathetic medical advice or report analyses based EXCLUSIVELY on the provided context.

Structure your response using Markdown:
1. **Executive Summary:** A 1-2 sentence direct answer.
2. **Detailed Clinical Analysis:** Explain mechanisms, symptoms, reports, or treatments.
3. **Important Considerations:** Any caveats, risk factors, or "seek immediate care" warnings.
4. **Citations:** You MUST append inline citations formatted strictly as [Source: filename, Page: X].

Context Blocks:
{context}

User Query: 
{question}

Clinical Response:"""

RAG_PROMPT = PromptTemplate(template=CUSTOM_RAG_PROMPT_TEMPLATE, input_variables=["context", "question"])


# =====================================================================
# 2. Application Routes (API Endpoints)
# =====================================================================

@app.route("/", methods=["GET"])
def index():
    """Simple root route so hitting the base URL doesn't 404."""
    return jsonify({
        "status": "ok",
        "service": "Medical Chatbot API",
        "endpoints": ["/health", "/chat (POST)", "/upload (POST)"]
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check route to confirm the server + LLM are reachable."""
    return jsonify({
        "status": "ok",
        "llm_initialized": llm is not None,
        "bm25_chunks_loaded": len(bm25_docs),
    })


@app.route("/chat", methods=["POST"])
def chat():
    """Main inference route called by Lovable UI with Routing, History, and Guardrails"""
    data = request.get_json()
    user_input = data.get("msg", "").strip()
    chat_history = data.get("chat_history", [])  # Expected format: [{"role": "user", "content": "..."}, ...]

    if not user_input:
        return jsonify({"error": "Empty query string provided."}), 400

    try:
        # --- FEATURE 1: History-Aware Contextualization ---
        search_query = user_input
        if chat_history:
            history_str = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in chat_history[-3:]])
            context_prompt = f"""Given the following chat history and new user question, rewrite the new question into a standalone, detailed medical search query.
            History:
            {history_str}
            New Question: {user_input}
            Standalone Query:"""
            search_query = llm_call(context_prompt).strip() or user_input

        # --- FEATURE 2: Intelligent Routing (LLM vs RAG) ---
        router_prompt = f"""Is the following user input a specific medical query/report analysis requiring database search, or just a general greeting/chit-chat? 
        Reply strictly with 'RAG' for medical, or 'LLM' for general greeting.
        User Input: {search_query}
        Decision:"""
        route_decision = llm_call(router_prompt).strip().upper() or "RAG"

        if "LLM" in route_decision and "RAG" not in route_decision:
            # Bypass heavy retrieval for simple greetings
            simple_response = llm_call(f"You are a polite clinical AI. Reply briefly to this: {search_query}")
            return jsonify({"response": simple_response})

        # --- FEATURE 3: Multi-Query Expansion ---
        multi_query_prompt = f"Generate 3 alternative medical variations (synonyms/clinical terms) of this query. Original Query: {search_query}"
        expanded_response = llm_call(multi_query_prompt).strip()
        search_queries = [search_query] + [q.strip() for q in expanded_response.split("\n") if q.strip()]

        # Retrieve & Accumulate (Hybrid RAG)
        retrieved_docs = []
        seen_contents = set()
        for query in search_queries:
            docs = ensemble_retriever.invoke(query)
            for doc in docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    retrieved_docs.append(doc)

        # --- FEATURE 4: Cross-Encoder Reranking ---
        final_reranked_docs = reranker.compress_documents(retrieved_docs, search_query)

        context_string = ""
        for i, doc in enumerate(final_reranked_docs):
            source = os.path.basename(doc.metadata.get("source", "Unknown Document"))
            page = doc.metadata.get("page", "N/A")
            context_string += f"\n--- Document {i+1} [Source: {source}, Page: {page}] ---\n{doc.page_content}\n"

        # Final Generation
        final_prompt = RAG_PROMPT.format(context=context_string, question=search_query)
        complete_answer = llm_call(final_prompt).strip() or "Mock Answer: LLM not initialized."

        # --- FEATURE 5: Safety Guardrails ---
        guardrail_prompt = f"""Review this AI answer based on the context provided. Does the AI hallucinate medical facts not present in the context, or provide dangerous prescriptive advice? 
        Context: {context_string}
        Answer: {complete_answer}
        Reply strictly 'PASS' if safe and factual, or 'REJECT' if unsafe or hallucinated."""

        validation = llm_call(guardrail_prompt).strip().upper() or "PASS"
        if "REJECT" in validation:
            return jsonify({"response": "For your safety, I cannot provide an unverified medical answer. Please consult a licensed physician regarding this query.", "guardrail_triggered": True})

        return jsonify({"response": complete_answer})

    except Exception as e:
        print(f"❌ Inference Error: {str(e)}")
        return jsonify({"error": "Internal processing error."}), 500


@app.route("/upload", methods=["POST"])
def upload_data():
    """Dynamic Ingestion for PDFs, Images (OCR), and Web Links without server restarts"""
    global bm25_retriever, bm25_docs, ensemble_retriever

    # 1. Handle Web Link Ingestion
    link = request.form.get("link")
    if link:
        try:
            print(f"🌐 Scraping data from: {link}")
            web_loader = WebBaseLoader(link)
            web_docs = web_loader.load()
            chunks = fast_splitter.split_documents(web_docs)

            PineconeVectorStore.from_documents(chunks, embeddings, index_name=INDEX_NAME)
            bm25_docs.extend(chunks)
            with open(BM25_CACHE_FILE, "wb") as f:
                pickle.dump(bm25_docs, f)
            bm25_retriever = BM25Retriever.from_documents(bm25_docs)
            ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, pinecone_retriever], weights=[0.5, 0.5])

            return jsonify({"message": f"Successfully learned from link: {link}"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Handle Files (Images and PDFs)
    if 'file' in request.files:
        file = request.files['file']

        # 2. Handle PDF Upload (Reports/Clinical Data)
        if file.filename.lower().endswith('.pdf'):
            try:
                print(f"📄 Processing dynamic PDF upload: {file.filename}")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    file.save(temp_pdf.name)
                    pdf_loader = PyPDFLoader(temp_pdf.name)
                    uploaded_docs = pdf_loader.load()

                    for doc in uploaded_docs:
                        doc.metadata["source"] = file.filename

                    chunks = fast_splitter.split_documents(uploaded_docs)

                    PineconeVectorStore.from_documents(chunks, embeddings, index_name=INDEX_NAME)
                    bm25_docs.extend(chunks)
                    with open(BM25_CACHE_FILE, "wb") as f:
                        pickle.dump(bm25_docs, f)
                    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
                    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, pinecone_retriever], weights=[0.5, 0.5])

                os.remove(temp_pdf.name)  # Clean up
                return jsonify({"message": f"Successfully processed PDF report: {file.filename}"}), 200
            except Exception as e:
                return jsonify({"error": f"PDF Processing Failed: {str(e)}"}), 500

        # 3. Handle Image/Prescription OCR Upload (Safe Pixel Handling)
        elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                print(f"👁️ Extracting text via OCR from image: {file.filename}")
                image = Image.open(file)

                if image.mode != 'RGB':
                    image = image.convert('RGB')

                image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)

                extracted_text = pytesseract.image_to_string(image)

                doc = Document(page_content=f"EXTRACTED MEDICAL IMAGE TEXT:\n{extracted_text}", metadata={"source": file.filename})
                chunks = fast_splitter.split_documents([doc])

                PineconeVectorStore.from_documents(chunks, embeddings, index_name=INDEX_NAME)
                bm25_docs.extend(chunks)
                with open(BM25_CACHE_FILE, "wb") as f:
                    pickle.dump(bm25_docs, f)
                bm25_retriever = BM25Retriever.from_documents(bm25_docs)
                ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, pinecone_retriever], weights=[0.5, 0.5])

                return jsonify({"message": f"Successfully analyzed and stored prescription image: {file.filename}"}), 200
            except Exception as e:
                return jsonify({"error": f"OCR Processing Failed: {str(e)}"}), 500

    return jsonify({"error": "No valid link, PDF, or image provided."}), 400


if __name__ == "__main__":
    # use_reloader=False stops Flask's debug reloader from re-importing this whole
    # script (and re-running all the model/Pinecone/BM25 init) a second time.
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)
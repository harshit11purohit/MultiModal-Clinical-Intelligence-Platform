import os
import time
from uuid import uuid4

from dotenv import load_dotenv
from tqdm.auto import tqdm
import tiktoken

from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.logger import logging
from src.utils import load_pdf, download_hugging_face_embeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "langchain-medical-chatbot"
TARGET_DIMENSION = 768  # BAAI/bge-base-en-v1.5 dimension size

# Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)

logging.info("Checking Pinecone index structure...")

# --- FORCE INDEX RECONSTRUCTION ON DIMENSION MISMATCH ---
existing_indexes = [idx.name for idx in pc.list_indexes()]

if INDEX_NAME in existing_indexes:
    desc = pc.describe_index(INDEX_NAME)
    if desc.dimension != TARGET_DIMENSION:
        logging.info(f"Dimension mismatch found (Index has {desc.dimension}, Model wants {TARGET_DIMENSION}). Recreating index...")
        pc.delete_index(INDEX_NAME)
        time.sleep(5)  # Give Pinecone a brief moment to clear cache
        existing_indexes.remove(INDEX_NAME)

# Create the correct hybrid index configuration
if INDEX_NAME not in existing_indexes:
    logging.info(f"Creating pristine serverless index: {INDEX_NAME} (Dimension: {TARGET_DIMENSION})")
    pc.create_index(
        name=INDEX_NAME,
        dimension=TARGET_DIMENSION,
        metric="dotproduct",  # Crucial for Hybrid Search scaling
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)

# Safely declare index hook variable now that we are guaranteed alignment
index = pc.Index(INDEX_NAME)

# --- DATA PARSING & CHUNKING ---
logging.info("Loading PDF documents...")
documents = load_pdf("data/")
logging.info(f"Loaded {len(documents)} source pages")

logging.info("Initialising BGE Embeddings...")
embeddings = download_hugging_face_embeddings()

tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    return len(tokenizer.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=tiktoken_len
)

texts = []
metadatas = []

for page in documents:
    chunks = text_splitter.split_text(page.page_content)
    for chunk_id, chunk in enumerate(chunks):
        texts.append(chunk)
        metadatas.append(
            {
                "source": page.metadata.get("source"),
                "page": page.metadata.get("page"),
                "chunk": chunk_id,
                "text": chunk,
            }
        )

# Fit Sparse Matrix
logging.info("Fitting Sparse BM25 Encoder...")
bm25 = BM25Encoder()
bm25.fit(texts)
bm25.dump("bm25_values.json")
logging.info("Saved BM25 matrices locally to bm25_values.json")

# Clear out lingering elements inside the index space
logging.info("Flushing index elements...")
try:
    index.delete(delete_all=True)
    time.sleep(2)
except Exception as e:
    pass

# --- UPLOAD BATCHES ---
BATCH_SIZE = 100
logging.info(f"Total Chunks: {len(texts)}")
logging.info("Uploading hybrid vectors...")

for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Uploading"):
    end = min(start + BATCH_SIZE, len(texts))
    
    batch_texts = texts[start:end]
    batch_metadata = metadatas[start:end]
    batch_ids = [str(uuid4()) for _ in range(len(batch_texts))]
    
    batch_dense_embeddings = embeddings.embed_documents(batch_texts)
    batch_sparse_embeddings = bm25.encode_documents(batch_texts)
    
    vectors = []
    for idx in range(len(batch_texts)):
        vectors.append(
            {
                "id": batch_ids[idx],
                "values": batch_dense_embeddings[idx],
                "sparse_values": batch_sparse_embeddings[idx],
                "metadata": batch_metadata[idx]
            }
        )
    
    index.upsert(vectors=vectors)

logging.info("Upload completed successfully.")
stats = index.describe_index_stats()

print("\n========================")
print("INDEX STATS")
print("========================")
print(stats)
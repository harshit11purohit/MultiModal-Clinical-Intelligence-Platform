# update_index.py
import os
from uuid import uuid4
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from src.utils import load_pdf, download_hugging_face_embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

load_dotenv()
INDEX_NAME = "langchain-medical-chatbot"


new_documents = load_pdf("data/new_updates/") 
if not new_documents:
    print("No new documents found to append.")
    exit()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)
embeddings = download_hugging_face_embeddings()

# 2. Chunk text targets
tokenizer = tiktoken.get_encoding("cl100k_base")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=150, length_function=lambda t: len(tokenizer.encode(t))
)

new_texts = []
new_metadatas = []
for page in new_documents:
    chunks = text_splitter.split_text(page.page_content)
    for chunk_id, chunk in enumerate(chunks):
        new_texts.append(chunk)
        new_metadatas.append({
            "source": page.metadata.get("source"),
            "page": page.metadata.get("page"),
            "chunk": chunk_id,
            "text": chunk,
        })

# 3. Load existing BM25 weights, update them, and re-save
bm25 = BM25Encoder()
if os.path.exists("bm25_values.json"):
    bm25.load("bm25_values.json")
    print("Loaded existing BM25 matrix. Updating with new vocabulary traits...")

bm25.fit(new_texts) # Update sparse matrix with new vocabulary variations
bm25.dump("bm25_values.json")

# 4. Upsert (Append) to Pinecone Index without clearing old entries
BATCH_SIZE = 100
for start in tqdm(range(0, len(new_texts), BATCH_SIZE), desc="Appending New Vectors"):
    end = min(start + BATCH_SIZE, len(new_texts))
    batch_texts = new_texts[start:end]
    batch_metadata = new_metadatas[start:end]
    batch_ids = [str(uuid4()) for _ in range(len(batch_texts))]
    
    batch_dense = embeddings.embed_documents(batch_texts)
    batch_sparse = bm25.encode_documents(batch_texts)
    
    vectors = []
    for idx in range(len(batch_texts)):
        vectors.append({
            "id": batch_ids[idx],
            "values": batch_dense[idx],
            "sparse_values": batch_sparse[idx],
            "metadata": batch_metadata[idx]
        })
    index.upsert(vectors=vectors)

print(f"Successfully appended {len(new_texts)} chunks into the index!")
import os
import json
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_pinecone import PineconeVectorStore

from src.utils import download_hugging_face_embeddings

load_dotenv()

INDEX_NAME = "langchain-medical-chatbot"

PROGRESS_FILE = "upload_progress.json"

BATCH_SIZE = 100


def save_progress(batch_no):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"batch": batch_no}, f)


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f).get("batch", 0)
    return 0


def reset_progress():
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)


def build_database():

    print("Loading Embedding Model...")
    embeddings = download_hugging_face_embeddings()

    print("Reading PDFs...")
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    if len(documents) == 0:
        print("No PDFs Found")
        return

    print("Semantic Chunking...")

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )

    chunks = splitter.split_documents(documents)

    print(f"Total Chunks : {len(chunks)}")

    db = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    start_batch = load_progress()

    print(f"Resuming From Batch {start_batch}")

    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch in range(start_batch, total_batches):

        start = batch * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(chunks))

        current = chunks[start:end]

        print(
            f"Uploading Batch {batch+1}/{total_batches}"
            f" ({len(current)} chunks)"
        )

        db.add_documents(current)

        save_progress(batch + 1)

    print("Upload Completed Successfully")

    reset_progress()


if __name__ == "__main__":
    build_database()
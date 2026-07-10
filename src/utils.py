from src.logger import logging

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader

from langchain_huggingface import HuggingFaceEmbeddings


def load_pdf(pdf_path):
    logging.info("Loading PDF files...")

    loader = DirectoryLoader(
        pdf_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    logging.info(f"Loaded {len(documents)} pages successfully.")

    return documents


def download_hugging_face_embeddings():
    logging.info("Loading BGE Base Embedding Model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},   # change to "cuda" if using GPU
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    logging.info("Embedding model loaded successfully.")

    return embeddings
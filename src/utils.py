from src.logger import *
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings


def load_pdf(pdf_path):
    logging.info("LOADING PDF FILE")
    loader=DirectoryLoader(pdf_path,glob='*.pdf',loader_cls=PyPDFLoader)
    document=loader.load()
    logging.info("file read successfully")
    return document


def download_embeddings():
    logging.info("downloading embedding")
    embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    logging.info("embedding downloaded")
    return embedding
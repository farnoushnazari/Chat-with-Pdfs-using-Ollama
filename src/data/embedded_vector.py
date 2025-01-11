import os
from typing import List
import ollama
from PyPDF2 import PdfReader
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from src.core.utils.config import cfg

def get_pdf_text(
        uploded_files: List[UploadedFile]
        ) -> str:
    text = ""
    for pdf in uploded_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunk(
        pdf_text: str
        ) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
        )
    texts = text_splitter.split_text(pdf_text)
    return texts

def get_vectorstore(
        texts: List[str]
        ) -> Chroma:
    ollama.pull(cfg['embedding_model'])

    embeddings = OllamaEmbeddings(model=cfg['embedding_model'])

    persist_path = "src/core/vector_db/"
    if not os.path.exists(persist_path):
        os.makedirs(persist_path)

    vector_db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=persist_path
        )
    vector_db.persist()

    return vector_db
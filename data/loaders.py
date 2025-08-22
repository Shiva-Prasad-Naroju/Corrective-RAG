from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import DOC_URLS

def load_documents():
    docs = [WebBaseLoader(url).load() for url in DOC_URLS]
    return [item for sublist in docs for item in sublist]

def split_documents(docs, chunk_size=500, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)

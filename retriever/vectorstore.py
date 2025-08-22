import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

VECTORSTORE_PATH = "vectorstore_index"

def build_vectorstore(doc_splits, persist=True):
    vectorstore = FAISS.from_documents(
        documents=doc_splits,
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    if persist:
        vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore

def load_vectorstore():
    if os.path.exists(VECTORSTORE_PATH):        
        return FAISS.load_local(VECTORSTORE_PATH, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),allow_dangerous_deserialization=True)
    return None

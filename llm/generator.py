from langchain import hub
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from config import GROQ_API_KEY

def get_rag_chain():
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, api_key=GROQ_API_KEY)
    prompt = hub.pull("rlm/rag-prompt")
    return prompt | llm | StrOutputParser()

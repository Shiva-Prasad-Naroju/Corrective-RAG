from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from config import GROQ_API_KEY

def get_question_rewriter():
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, api_key=GROQ_API_KEY)
    
    system = """You are a question re-writer that converts a question 
    into a version optimized for web search, preserving semantic intent."""
    
    re_write_prompt = ChatPromptTemplate.from_messages(
        [("system", system),
         ("human", "Here is the initial question: \n\n {question} \n Improve it.")]
    )
    
    return re_write_prompt | llm | StrOutputParser()

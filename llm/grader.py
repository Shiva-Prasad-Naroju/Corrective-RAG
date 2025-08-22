from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from config import GROQ_API_KEY

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="yes/no if relevant")

def get_retrieval_grader():
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, api_key=GROQ_API_KEY)

    structured_llm = llm.with_structured_output(GradeDocuments, strict=True)

    system_prompt = """You are a grader assessing relevance of a retrieved document to a user question.
    Only return {{ "binary_score": "yes" }} or {{ "binary_score": "no" }}.
    Do not return anything else."""

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}")
    ])

    return grade_prompt | structured_llm

# ----- CORRECTIVE RAG -----
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY=os.getenv("GROQ_API_KEY")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----- Docs to index -----
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# ----- Load -----
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# ----- Splitting the chunks -----
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

# ----- Add to vectorstore -----
vectorstore=FAISS.from_documents(
    documents=doc_splits,
    embedding=embeddings
)
retriever = vectorstore.as_retriever()

# ----- Retrieval Grader -----
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

# ----- Data Model -----
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# ----- LLM with function call -----
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, api_key=GROQ_API_KEY)
structured_llm_grader = llm.with_structured_output(GradeDocuments) # Outputs relevance score Yes or No

# ----- Prompt -----
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# ----- Chain the prompt with the LLM -----
retrieval_grader = grade_prompt | structured_llm_grader

question = "AI Agents"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content

# print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

# ----- Generate -----
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = llm

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = prompt | llm | StrOutputParser()

# Run
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)

# ----- Question Re-writer -----

# LLM
llm = llm

# Prompt
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)
question_rewriter = re_write_prompt | llm | StrOutputParser()
question_rewriter.invoke({"question": question})

# ----- Search -----
from langchain_community.tools import DuckDuckGoSearchResults
web_search_tool = DuckDuckGoSearchResults(k=3)

from typing import List
from typing_extensions import TypedDict

# ----- State -----
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]

from langchain.schema import Document

# ----- Retrieve function -----
def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


# ----- Generate function -----
def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


# ----- Grade document function -----
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


# ----- Transform / Rewrite Query function -----
def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


# ----- Web Search function -----
def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})

    # Handle both string and list-of-dicts results
    if isinstance(docs, str):
        # Single string result
        web_results = Document(page_content=docs)
    elif isinstance(docs, list):
        # Multi-result list of dicts
        web_text = "\n".join([d.get("content", "") for d in docs])
        web_results = Document(page_content=web_text)
    else:
        # Fallback if the tool returns something unexpected
        web_results = Document(page_content=str(docs))

    documents.append(web_results)

    return {"documents": documents, "question": question}



### Edges

# ----- Decides to Generate or Rewrite the query - Function -----
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    

# ----- Graph Building -----
from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)                 # retrieve
workflow.add_node("grade_documents", grade_documents)   # grade documents
workflow.add_node("generate", generate)                 # generate
workflow.add_node("transform_query", transform_query)   # transform_query
workflow.add_node("web_search_node", web_search)        # web search

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()


# ----- Displaying the workflow image -----
from IPython.display import Image, display
display(Image(app.get_graph(xray=True).draw_mermaid_png()))

# ----- Invoking the workflow -----
app.invoke({"question":"What are the types of agent memory?"})


def print_readable_output(output):
    print("\n--- QUESTION ---")
    print(output.get("question", ""))

    print("\n--- GENERATION ---")
    print(output.get("generation", ""))

    print("\n--- WEB SEARCH STATUS ---")
    print(output.get("web_search", ""))

    print("\n--- DOCUMENTS ---")
    documents = output.get("documents", [])
    for i, doc in enumerate(documents, 1):
        print(f"\nDocument {i}:")
        # Print metadata if available
        if hasattr(doc, "metadata") and doc.metadata:
            for k, v in doc.metadata.items():
                print(f"  {k}: {v}")
        # Print page content
        if hasattr(doc, "page_content"):
            snippet = doc.page_content[:500]  # print first 500 chars for readability
            print(f"  page_content (snippet):\n{snippet}\n  ...")
        else:
            print(f"  {doc}")  # fallback



# ----- Printing in the human readable format -----            
print_readable_output(app.invoke({"question":"What are the types of agent memory?"}))












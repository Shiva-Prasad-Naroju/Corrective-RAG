from data.loaders import load_documents, split_documents
from retriever.vectorstore import build_vectorstore, load_vectorstore
from retriever.retrieval import get_retriever
from llm.grader import get_retrieval_grader
from llm.generator import get_rag_chain
from llm.rewriter import get_question_rewriter
from workflow.graph import build_workflow
from utils.formatting import print_readable_output

def main():
    # Load or build vectorstore
    print("---Building / Loading the Vectorstore---")
    vectorstore = load_vectorstore()
    if vectorstore is None:
        print("No saved vectorstore found, building from scratch...")
        docs = load_documents()
        doc_splits = split_documents(docs)
        vectorstore = build_vectorstore(doc_splits)
    else:
        print("Loaded existing vectorstore from disk.")

    # Build retriever
    retriever = get_retriever(vectorstore)

    # LLM components
    retrieval_grader = get_retrieval_grader()
    rag_chain = get_rag_chain()
    question_rewriter = get_question_rewriter()

    # Build workflow
    print("---Building the Graph Workflow---")
    app = build_workflow(retriever, rag_chain, retrieval_grader, question_rewriter)

    # Run a sample query
    output = app.invoke({"question": "What are the types of agent memory?"})
    print_readable_output(output)

if __name__ == "__main__":
    main()

from langgraph.graph import END, StateGraph, START
from .state import GraphState
from .nodes import retrieve, generate, grade_documents, transform_query, web_search_node

def decide_to_generate(state):
    print("---ASSESS DOCUMENTS---")
    if state["web_search"] == "Yes":
        return "transform_query"
    return "generate"

def build_workflow(retriever, rag_chain, retrieval_grader, question_rewriter):
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", lambda s: retrieve(s, retriever))
    workflow.add_node("grade_documents", lambda s: grade_documents(s, retrieval_grader))
    workflow.add_node("generate", lambda s: generate(s, rag_chain))
    workflow.add_node("transform_query", lambda s: transform_query(s, question_rewriter))
    workflow.add_node("web_search_node", web_search_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate, {
        "transform_query": "transform_query",
        "generate": "generate"
    })
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

from utils.search import run_web_search

def retrieve(state, retriever):
    print("---RETRIEVE---")
    docs = retriever.invoke(state["question"])
    return {"documents": docs, "question": state["question"]}

def generate(state, rag_chain):
    print("---GENERATE---")
    generation = rag_chain.invoke({"context": state["documents"], "question": state["question"]})
    return {**state, "generation": generation}

def grade_documents(state, retrieval_grader):
    print("---GRADE DOCUMENTS---")
    filtered_docs, web_search = [], "No"
    for d in state["documents"]:
        score = retrieval_grader.invoke({"question": state["question"], "document": d.page_content})
        if score.binary_score == "yes":
            filtered_docs.append(d)
        else:
            web_search = "Yes"
    return {"documents": filtered_docs, "question": state["question"], "web_search": web_search}

def transform_query(state, question_rewriter):
    print("---TRANSFORM QUERY---")
    new_q = question_rewriter.invoke({"question": state["question"]})
    return {"documents": state["documents"], "question": new_q}

def web_search_node(state):
    print("---WEB SEARCH---")
    docs = run_web_search(state["question"], state["documents"])
    return {"documents": docs, "question": state["question"]}

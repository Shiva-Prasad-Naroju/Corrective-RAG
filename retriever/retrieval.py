def get_retriever(vectorstore):
    # retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5, "score_threshold": 0.5})
    # return retriever
    return vectorstore.as_retriever()





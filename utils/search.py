from ddgs import DDGS
from langchain.schema import Document

def run_web_search(question, documents):
    """
    Perform a DuckDuckGo web search using DDGS and append results to documents.
    """
    with DDGS() as ddgs:
        results = ddgs.text(query=question, max_results=3)
    
    # Collect text from results
    web_text = "\n".join([r.get('body', '') for r in results])
    
    # Wrap in a LangChain Document
    web_results = Document(page_content=web_text)
    
    documents.append(web_results)
    return documents

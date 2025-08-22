def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def print_readable_output(output):
    print("\n--- QUESTION ---")
    print(output.get("question", ""))

    print("\n--- GENERATION ---")
    print(output.get("generation", ""))

    print("\n--- WEB SEARCH STATUS ---")
    print(output.get("web_search", ""))

    print("\n--- DOCUMENTS ---")
    for i, doc in enumerate(output.get("documents", []), 1):
        print(f"\nDocument {i}: {getattr(doc, 'page_content', '')[:500]}...")

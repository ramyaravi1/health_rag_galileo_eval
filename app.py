from rag_cli import run_rag

if __name__ == "__main__":
    print("Healthcare RAG Chatbot (FLAN-T5)")
    print("Type 'exit' to quit\n")

    while True:
        query = input("Ask: ")
        if query.lower() == "exit":
            break

        answer = run_rag(query)
        print("\nAnswer:", answer, "\n")
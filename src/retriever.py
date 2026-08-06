from vector_store import create_vector_store
from text_splitter import split_documents
from document_loader import load_documents


def create_retriever():
    """
    Create FAISS retriever from documents.
    """

    documents = load_documents(
        "data/medical_documents"
    )

    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks)

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    return retriever

if __name__ == "__main__":

    retriever = create_retriever()

    question = "What are symptoms of diabetes?"

    results = retriever.invoke(question)

    print(f"Found {len(results)} documents\n")

    for i, doc in enumerate(results):
        print(f"Result {i+1}:")
        print(doc.page_content)
        print("-------------------")
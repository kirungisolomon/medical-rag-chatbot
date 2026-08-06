from langchain_community.vectorstores import FAISS
from embeddings import create_embeddings


def create_vector_store(documents):
    """
    Create FAISS vector database from documents.
    """

    embedding_model = create_embeddings()

    vector_store = FAISS.from_documents(
        documents,
        embedding_model
    )

    return vector_store


if __name__ == "__main__":
    print("FAISS vector store module ready")
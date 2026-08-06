from langchain_huggingface import HuggingFaceEmbeddings

def create_embeddings():
    """
    Create free Hugging Face embedding model.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


if __name__ == "__main__":

    embedding_model = create_embeddings()

    vector = embedding_model.embed_query(
        "What are symptoms of diabetes?"
    )

    print(f"Embedding dimensions: {len(vector)}")
    print(vector[:5])
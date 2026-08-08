from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings():
    """
    Create local Hugging Face embedding model.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings


if __name__ == "__main__":

    embedding_model = create_embeddings()

    vector = embedding_model.embed_query(
        "What are symptoms of diabetes?"
    )

    print(f"Embedding dimensions: {len(vector)}")
    print(vector[:5])
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    """
    Split documents into smaller chunks for embedding.

    Args:
        documents: List of LangChain Document objects

    Returns:
        Split document chunks
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":
    from document_loader import load_documents

    documents = load_documents("data/medical_documents")

    chunks = split_documents(documents)

    print(f"Original documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    if chunks:
        print("\nSample chunk:")
        print(chunks[0].page_content[:500])
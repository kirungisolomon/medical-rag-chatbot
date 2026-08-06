from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_documents(directory: str):
    """
    Load PDF documents from a directory.

    Args:
        directory: Path containing PDF files

    Returns:
        List of LangChain Document objects
    """

    documents = []

    pdf_files = Path(directory).glob("*.pdf")

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    return documents


if __name__ == "__main__":
    docs = load_documents("data/medical_documents")

    print(f"Loaded {len(docs)} document pages")

    if docs:
        print(docs[0].page_content[:500])
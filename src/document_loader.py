import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_documents(directory):
    """
    Load PDF and TXT medical documents.
    """

    documents = []

    for filename in os.listdir(directory):

        file_path = os.path.join(
            directory,
            filename
        )

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())

    return documents


if __name__ == "__main__":

    docs = load_documents(
        "data/medical_documents"
    )

    print(f"Loaded {len(docs)} document pages")
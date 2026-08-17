from pathlib import Path

from langchain_community.document_loaders import TextLoader


BASE_DIR = Path(__file__).resolve().parents[1]
DOCUMENT_PATH = BASE_DIR / "data" / "faq_document.txt"


def load_documents():
    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el documento fuente: {DOCUMENT_PATH}"
        )

    return TextLoader(
        str(DOCUMENT_PATH),
        encoding="utf-8",
    ).load()

# testing
# if __name__ == "__main__":
#     documents = load_documents()
#
#     print("Cantidad de documentos:", len(documents))
#
#     for document in documents:
#         print("\nMETADATA:")
#         print(document.metadata)
#
#         print("\nCONTENIDO:")
#         print(document.page_content[:500])

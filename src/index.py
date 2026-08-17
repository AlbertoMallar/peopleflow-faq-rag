from src.chunking import (
    split_into_chunks,
    split_sections,
    validate_chunks,
)
from src.document_loader import load_documents
from src.vector_store import build_vector_store


def build_index():
    documents = load_documents()
    sections = split_sections(documents)
    chunks = split_into_chunks(sections)

    validate_chunks(chunks)

    vector_store = build_vector_store(chunks)

    print(f"Documentos: {len(documents)}")
    print(f"Secciones: {len(sections)}")
    print(f"Chunks indexados: {len(chunks)}")
    print("Índice vectorial creado correctamente.")

    return vector_store


if __name__ == "__main__":
    build_index()
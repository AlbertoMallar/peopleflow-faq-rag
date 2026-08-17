import hashlib
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.embeddings import get_embedding_model


BASE_DIR = Path(__file__).resolve().parents[1]
PERSIST_DIRECTORY = BASE_DIR / "storage" / "chroma"
COLLECTION_NAME = "peopleflow_faq"
COLLECTION_CONFIGURATION = {
    "hnsw": {
        "space": "l2",
    }
}


def generate_chunk_id(chunk: Document) -> str:
    content = chunk.page_content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


def attach_chunk_ids(chunks: list[Document]) -> list[str]:
    ids = []

    for chunk in chunks:
        chunk_id = generate_chunk_id(chunk)
        chunk.metadata["chunk_id"] = chunk_id
        ids.append(chunk_id)

    return ids


def build_vector_store(chunks: list[Document]) -> Chroma:
    if PERSIST_DIRECTORY.exists():
        shutil.rmtree(PERSIST_DIRECTORY)

    ids = attach_chunk_ids(chunks)

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        ids=ids,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIRECTORY),
        collection_configuration=COLLECTION_CONFIGURATION,
    )


def load_vector_store() -> Chroma:
    if not PERSIST_DIRECTORY.exists():
        raise FileNotFoundError(
            "No existe un indice persistido en storage/chroma. "
            "Ejecuta src/build_index.py primero."
        )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=str(PERSIST_DIRECTORY),
        collection_configuration=COLLECTION_CONFIGURATION,
        create_collection_if_not_exists=False,
    )


def get_vector_store_info(vector_store: Chroma | None = None) -> dict[str, object]:
    vector_store = vector_store or load_vector_store()
    collection = vector_store._collection
    configuration = collection.configuration or {}
    hnsw_config = configuration.get("hnsw") or {}

    return {
        "collection_name": collection.name,
        "document_count": collection.count(),
        "distance_space": hnsw_config.get("space"),
        "persist_directory": str(PERSIST_DIRECTORY),
    }

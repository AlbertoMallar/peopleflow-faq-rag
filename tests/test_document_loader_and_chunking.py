from pathlib import Path

from src.chunking import count_tokens, validate_chunks
from src.vector_store import generate_chunk_id


def test_load_documents_carga_el_corpus_principal(documents):
    assert len(documents) == 1
    document = documents[0]

    assert Path(document.metadata["source"]).name == "faq_document.txt"
    assert len(document.page_content.split()) >= 1000


def test_chunking_generates_at_least_20_chunks(chunks):
    assert len(chunks) >= 20


def test_todos_los_chunks_quedan_entre_50_y_500_tokens(chunks):
    validate_chunks(chunks)
    token_counts = [count_tokens(chunk.page_content) for chunk in chunks]

    assert min(token_counts) >= 50
    assert max(token_counts) <= 500


def test_metadata_importante_se_conserva(chunks):
    first_chunk = chunks[0]
    second_chunk = chunks[1]

    for chunk in (first_chunk, second_chunk):
        assert "source" in chunk.metadata
        assert "document_title" in chunk.metadata
        assert "part" in chunk.metadata
        assert "total_parts" in chunk.metadata
        assert "token_count" in chunk.metadata

    assert "section_title" in second_chunk.metadata


def test_chunk_ids_son_deterministas(chunks):
    first_run = [generate_chunk_id(chunk) for chunk in chunks]
    second_run = [generate_chunk_id(chunk) for chunk in chunks]

    assert first_run == second_run

from langchain_core.documents import Document

from src.query import answer_question, build_context


def make_chunk(chunk_id: str, section_title: str, content: str, score: float) -> Document:
    return Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "section_title": section_title,
            "relevance_score": score,
        },
    )


def test_build_context_incluye_chunk_id_y_seccion():
    chunks = [
        make_chunk("abc123", "Vacaciones", "Contenido A", 0.9),
        make_chunk("def456", "Permisos", "Contenido B", 0.8),
    ]

    context = build_context(chunks)

    assert "[Chunk abc123 | Sección: Vacaciones]" in context
    assert "Contenido B" in context


def test_answer_question_con_mocks(monkeypatch):
    chunks = [
        make_chunk("chunk-1", "Vacaciones", "Paso 1", 0.91),
        make_chunk("chunk-2", "Saldo de vacaciones y ajustes", "Paso 2", 0.82),
        make_chunk("chunk-3", "Otras ausencias y permisos", "Paso 3", 0.77),
    ]

    monkeypatch.setattr("src.query.retrieve_chunks", lambda question: chunks)
    monkeypatch.setattr(
        "src.query.generate_answer",
        lambda question, context: "Pedí vacaciones desde la sección Vacaciones.",
    )

    response = answer_question(" ¿Cómo solicito vacaciones? ")
    payload = response.model_dump()

    assert payload["user_question"] == "¿Cómo solicito vacaciones?"
    assert payload["system_answer"] == "Pedí vacaciones desde la sección Vacaciones."
    assert list(payload.keys()) == [
        "user_question",
        "system_answer",
        "chunks_related",
    ]
    assert len(payload["chunks_related"]) == 3

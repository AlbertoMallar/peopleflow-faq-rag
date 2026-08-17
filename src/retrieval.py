from langchain_core.documents import Document

from src.vector_store import load_vector_store


TOP_K = 3
MIN_TOP_K = 2
MAX_TOP_K = 5


def validate_question(question: str) -> str:
    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError("La pregunta no puede estar vacía.")

    return normalized_question


def validate_top_k(top_k: int) -> int:
    if not MIN_TOP_K <= top_k <= MAX_TOP_K:
        raise ValueError(
            f"top_k debe estar entre {MIN_TOP_K} y {MAX_TOP_K}."
        )

    return top_k


def attach_relevance_scores(
    results: list[tuple[Document, float]],
) -> list[Document]:
    retrieved_chunks = []

    for document, score in results:
        document.metadata["relevance_score"] = round(float(score), 4)
        retrieved_chunks.append(document)

    return retrieved_chunks


def retrieve_chunks(
    question: str,
    top_k: int = TOP_K,
) -> list[Document]:
    normalized_question = validate_question(question)
    valid_top_k = validate_top_k(top_k)
    vector_store = load_vector_store()

    if vector_store._collection.count() == 0:
        raise ValueError(
            "El índice vectorial está vacío. Ejecutá src/build_index.py primero."
        )

    results = vector_store.similarity_search_with_relevance_scores(
        normalized_question,
        k=valid_top_k,
    )
    return attach_relevance_scores(results)


if __name__ == "__main__":
    questions = [
        "¿Cómo solicito vacaciones?",
        "¿Qué hago si pierdo el acceso al segundo factor de autenticación?",
        "¿Cómo puedo importar empleados mediante un archivo CSV?",
        "¿PeopleFlow permite registrar asistencia mediante reconocimiento facial?",
    ]

    for question in questions:
        print(f"\nPREGUNTA: {question}")

        chunks = retrieve_chunks(question)

        for index, chunk in enumerate(chunks, start=1):
            title = chunk.metadata.get(
                "section_title",
                chunk.metadata.get("document_title", "Sin título"),
            )

            print(
                f"{index}. {title} | "
                f"score: {chunk.metadata['relevance_score']}"
            )

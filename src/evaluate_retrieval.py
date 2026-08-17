import json
from dataclasses import dataclass
from statistics import mean

from src.retrieval import TOP_K, retrieve_chunks


@dataclass(frozen=True)
class RetrievalCase:
    question: str
    relevant_sections: tuple[str, ...]


EVALUATION_CASES = [
    RetrievalCase(
        question="¿Cómo solicito vacaciones?",
        relevant_sections=(
            "Vacaciones",
            "Otras ausencias y permisos",
            "Saldo de vacaciones y ajustes",
        ),
    ),
    RetrievalCase(
        question="¿Qué hago si pierdo el acceso al segundo factor de autenticación?",
        relevant_sections=(
            "Autenticación multifactor",
            "Acceso y contraseñas",
            "Seguridad y auditoría",
        ),
    ),
    RetrievalCase(
        question="¿Cómo puedo importar empleados mediante un archivo CSV?",
        relevant_sections=(
            "Importación masiva por CSV",
            "Exportaciones",
        ),
    ),
    RetrievalCase(
        question="¿Puede un líder modificar el saldo de vacaciones de su equipo?",
        relevant_sections=(
            "Saldo de vacaciones y ajustes",
            "Vacaciones",
        ),
    ),
]


def extract_section_titles(chunks) -> list[str]:
    titles = []

    for chunk in chunks:
        titles.append(
            chunk.metadata.get(
                "section_title",
                chunk.metadata.get("document_title", "Sin título"),
            )
        )

    return titles


def precision_at_k(
    retrieved_titles: list[str],
    relevant_sections: tuple[str, ...],
    k: int,
) -> float:
    hits = sum(
        1 for title in retrieved_titles[:k]
        if title in relevant_sections
    )
    return round(hits / k, 4)


def hit_at_k(
    retrieved_titles: list[str],
    relevant_sections: tuple[str, ...],
    k: int,
) -> int:
    return int(
        any(title in relevant_sections for title in retrieved_titles[:k])
    )


def evaluate_case(case: RetrievalCase) -> dict[str, object]:
    chunks = retrieve_chunks(case.question, top_k=TOP_K)
    titles = extract_section_titles(chunks)

    return {
        "question": case.question,
        "relevant_sections": list(case.relevant_sections),
        "retrieved_sections": titles,
        "precision_at_3": precision_at_k(titles, case.relevant_sections, TOP_K),
        "hit_at_3": hit_at_k(titles, case.relevant_sections, TOP_K),
    }


def evaluate_retrieval() -> dict[str, object]:
    if TOP_K != 3:
        raise ValueError("La evaluación reproducible espera TOP_K=3.")

    case_results = [evaluate_case(case) for case in EVALUATION_CASES]
    average_precision = round(
        mean(result["precision_at_3"] for result in case_results),
        4,
    )

    return {
        "top_k": TOP_K,
        "average_precision_at_3": average_precision,
        "cases": case_results,
    }


def main() -> None:
    result = evaluate_retrieval()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

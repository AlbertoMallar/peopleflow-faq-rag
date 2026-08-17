import json
from pathlib import Path

from src.query import answer_question


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BASE_DIR / "outputs" / "sample_queries.json"

SAMPLE_QUESTIONS = [
    "¿Cómo solicito vacaciones?",
    "¿Qué hago si pierdo el acceso al segundo factor de autenticación?",
    "¿Cómo puedo importar empleados mediante un archivo CSV?",
    "¿Puede un líder modificar el saldo de vacaciones de su equipo?",
]


def generate_samples() -> None:
    results = []

    for question in SAMPLE_QUESTIONS:
        response = answer_question(question)
        results.append(response.model_dump())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Ejemplos generados: {len(results)}")
    print(f"Archivo creado: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_samples()
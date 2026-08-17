import pytest
from langchain_core.documents import Document

from src.evaluator import (
    EvaluationResult,
    evaluate_response,
    validate_evaluation,
)


def test_validate_evaluation_acepta_payload_valido():
    result = validate_evaluation(
        {
            "score": 8,
            "reason": (
                "Los chunks son pertinentes, la respuesta se mantiene fiel "
                "a la evidencia y cubre bien la pregunta del usuario."
            ),
        }
    )

    assert isinstance(result, EvaluationResult)
    assert result.score == 8


def test_validate_evaluation_rechaza_score_fuera_de_rango():
    with pytest.raises(ValueError, match="less than or equal to 10"):
        validate_evaluation(
            {
                "score": 11,
                "reason": (
                    "La justificación supera el mínimo requerido, pero el "
                    "score no respeta el rango permitido por la rúbrica."
                ),
            }
        )


def test_validate_evaluation_rechaza_reason_corto():
    with pytest.raises(ValueError, match="al menos 50 caracteres"):
        validate_evaluation(
            {
                "score": 6,
                "reason": "Muy corto.",
            }
        )


def test_evaluate_response_con_llm_mockeado(monkeypatch):
    class FakeChain:
        @staticmethod
        def invoke(payload):
            assert payload["question"] == "¿Cómo solicito vacaciones?"
            assert "Vacaciones" in payload["chunks"]
            return {
                "score": 9,
                "reason": (
                    "Los chunks recuperados son relevantes, la respuesta se apoya "
                    "en ellos y contesta la consulta de manera suficiente."
                ),
            }

    monkeypatch.setattr("src.evaluator.get_evaluator_chain", lambda: FakeChain())

    result = evaluate_response(
        question="¿Cómo solicito vacaciones?",
        answer="Ingresá a la sección Vacaciones y enviá la solicitud.",
        chunks=[
            Document(
                page_content="Ingresá a Vacaciones y seleccioná las fechas.",
                metadata={
                    "chunk_id": "chunk-1",
                    "section_title": "Vacaciones",
                },
            )
        ],
    )

    assert result.score == 9


def test_evaluate_response_rechaza_respuesta_vacia():
    with pytest.raises(ValueError, match="no puede estar vacía"):
        evaluate_response(
            question="¿Cómo solicito vacaciones?",
            answer="   ",
            chunks=[],
        )

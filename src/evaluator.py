import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.retrieval import validate_question
from src.schemas import ChunkRelated, RAGResponse


load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL")

if not LLM_MODEL:
    raise ValueError("Falta definir LLM_MODEL en el archivo .env")

EVALUATOR_PROMPT_TEMPLATE = """Evaluá la calidad de una respuesta RAG.

Tenés que puntuar de 0 a 10 y justificar en castellano, con al menos 50 caracteres.
La evaluación debe considerar como mínimo:
1. relevancia de los chunks respecto de la pregunta;
2. grounding o fidelidad de la respuesta respecto de los chunks;
3. completitud de la respuesta respecto de la pregunta.

Respondé únicamente con la estructura pedida.

PREGUNTA:
{question}

RESPUESTA DEL SISTEMA:
{answer}

CHUNKS RECUPERADOS:
{chunks}
"""


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=0, le=10)
    reason: str

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        normalized_reason = value.strip()

        if len(normalized_reason) < 50:
            raise ValueError(
                "reason debe tener al menos 50 caracteres."
            )

        return normalized_reason


def build_chunks_text(chunks: list[Document | ChunkRelated]) -> str:
    lines = []

    for index, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, Document):
            chunk_id = chunk.metadata.get("chunk_id", "sin-id")
            section_title = chunk.metadata.get(
                "section_title",
                chunk.metadata.get("document_title", "Sin título"),
            )
            content = chunk.page_content
        else:
            chunk_id = chunk.chunk_id
            section_title = chunk.section_title
            content = chunk.content

        lines.append(
            f"{index}. chunk_id={chunk_id} | sección={section_title}\n{content}"
        )

    return "\n\n".join(lines)


def get_evaluator_chain():
    prompt = ChatPromptTemplate.from_template(EVALUATOR_PROMPT_TEMPLATE)
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
    ).with_structured_output(EvaluationResult)
    return prompt | llm


def validate_evaluation(
    evaluation: EvaluationResult | dict,
) -> EvaluationResult:
    if isinstance(evaluation, EvaluationResult):
        return EvaluationResult.model_validate(evaluation.model_dump())

    return EvaluationResult.model_validate(evaluation)


def evaluate_response(
    question: str,
    answer: str,
    chunks: list[Document | ChunkRelated],
) -> EvaluationResult:
    normalized_question = validate_question(question)
    normalized_answer = answer.strip()

    if not normalized_answer:
        raise ValueError("La respuesta a evaluar no puede estar vacía.")

    chain = get_evaluator_chain()
    evaluation = chain.invoke(
        {
            "question": normalized_question,
            "answer": normalized_answer,
            "chunks": build_chunks_text(chunks),
        }
    )
    return validate_evaluation(evaluation)


def evaluate_rag_response(response: RAGResponse) -> EvaluationResult:
    return evaluate_response(
        question=response.user_question,
        answer=response.system_answer,
        chunks=response.chunks_related,
    )

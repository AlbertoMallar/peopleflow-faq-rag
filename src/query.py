import argparse
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.retrieval import retrieve_chunks, validate_question
from src.schemas import ChunkRelated, RAGResponse


load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL")

if not LLM_MODEL:
    raise ValueError("Falta definir LLM_MODEL en el archivo .env")

RAG_PROMPT_TEMPLATE = """Sos un asistente de soporte de PeopleFlow, una plataforma de RR.HH.

Respondé solamente con la información presente en el contexto recuperado.
No inventes datos, pasos ni funcionalidades y no uses conocimiento externo.
Respondé únicamente lo necesario para contestar la pregunta, de manera clara,
natural y breve.

Si el contexto no contiene información suficiente para responder, indicá:
"No tengo información suficiente en la documentación disponible."

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA:"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consulta el chatbot RAG de PeopleFlow."
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Pregunta del usuario.",
    )
    return parser.parse_args()


def build_context(chunks: list[Document]) -> str:
    context_parts = []

    for chunk in chunks:
        chunk_id = chunk.metadata.get("chunk_id", "sin-id")
        title = chunk.metadata.get(
            "section_title",
            chunk.metadata.get("document_title", "Sin título"),
        )
        context_parts.append(
            f"[Chunk {chunk_id} | Sección: {title}]\n"
            f"{chunk.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)


def get_generation_chain():
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    return prompt | ChatOpenAI(model=LLM_MODEL) | StrOutputParser()


def generate_answer(question: str, context: str) -> str:
    chain = get_generation_chain()
    return chain.invoke(
        {
            "context": context,
            "question": question,
        }
    ).strip()


def format_related_chunks(
    chunks: list[Document],
) -> list[ChunkRelated]:
    related_chunks = []

    for chunk in chunks:
        related_chunks.append(
            ChunkRelated(
                chunk_id=chunk.metadata.get("chunk_id", "sin-id"),
                section_title=chunk.metadata.get(
                    "section_title",
                    chunk.metadata.get("document_title", "Sin título"),
                ),
                relevance_score=chunk.metadata.get(
                    "relevance_score",
                    0.0,
                ),
                content=chunk.page_content,
            )
        )

    return related_chunks


def answer_question(question: str) -> RAGResponse:
    normalized_question = validate_question(question)
    chunks = retrieve_chunks(normalized_question)
    context = build_context(chunks)
    answer = generate_answer(normalized_question, context)

    return RAGResponse(
        user_question=normalized_question,
        system_answer=answer,
        chunks_related=format_related_chunks(chunks),
    )


def main() -> None:
    args = parse_args()
    result = answer_question(args.question)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

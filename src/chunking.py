import os

import tiktoken
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

if not EMBEDDING_MODEL:
    raise ValueError("Falta definir EMBEDDING_MODEL en el archivo .env")

TOKENIZER = tiktoken.encoding_for_model(EMBEDDING_MODEL)
HEADERS_TO_SPLIT_ON = [
    ("#", "document_title"),
    ("##", "section_title"),
]
MAX_SECTION_TOKENS = 300
SPLIT_CHUNK_SIZE = 200
CHUNK_OVERLAP = 30


def split_sections(documents: list[Document]) -> list[Document]:
    sections = []
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON
    )

    for document in documents:
        document_sections = splitter.split_text(document.page_content)

        for section in document_sections:
            section.metadata = {
                **document.metadata,
                **section.metadata,
            }
            sections.append(section)

    return sections


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


if False:  # Cambiar a True para volver a ejecutar el conteo de tokens.
    if __name__ == "__main__":
        from src.document_loader import load_documents

        documents = load_documents()
        sections = split_sections(documents)

        for section in sections:
            title = section.metadata.get(
                "section_title",
                section.metadata.get("document_title", "Sin título"),
            )
            token_count = count_tokens(section.page_content)
            print(f"{title}: {token_count} tokens")


def get_section_title(section: Document) -> str:
    return section.metadata.get(
        "section_title",
        section.metadata.get("document_title", ""),
    )


def build_chunk_content(title: str, content: str) -> str:
    return f"{title}\n\n{content}".strip()


def split_large_section(section: Document, title: str) -> list[Document]:
    title_tokens = count_tokens(title)
    content_chunk_size = SPLIT_CHUNK_SIZE - title_tokens

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKENIZER.name,
        chunk_size=content_chunk_size,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents([section])


def split_section_parts(section: Document) -> list[Document]:
    title = get_section_title(section)
    section_text = build_chunk_content(title, section.page_content)

    if count_tokens(section_text) <= MAX_SECTION_TOKENS:
        return [section]

    return split_large_section(section, title)


def build_chunk(
    part: Document,
    title: str,
    part_number: int,
    total_parts: int,
) -> Document:
    content = build_chunk_content(title, part.page_content)
    metadata = {
        **part.metadata,
        "part": part_number,
        "total_parts": total_parts,
        "token_count": count_tokens(content),
    }

    return Document(
        page_content=content,
        metadata=metadata,
    )


def split_into_chunks(sections: list[Document]) -> list[Document]:
    chunks = []

    for section in sections:
        title = get_section_title(section)
        parts = split_section_parts(section)

        for part_number, part in enumerate(parts, start=1):
            chunks.append(
                build_chunk(
                    part=part,
                    title=title,
                    part_number=part_number,
                    total_parts=len(parts),
                )
            )

    return chunks


def validate_chunks(chunks: list[Document]) -> None:
    for index, chunk in enumerate(chunks, start=1):
        token_count = count_tokens(chunk.page_content)

        if not 50 <= token_count <= 500:
            raise ValueError(
                f"Chunk {index} fuera del rango permitido: "
                f"{token_count} tokens"
            )


if __name__ == "__main__":
    from src.document_loader import load_documents

    documents = load_documents()
    sections = split_sections(documents)
    chunks = split_into_chunks(sections)

    validate_chunks(chunks)

    print("Documentos cargados:", len(documents))
    print("Secciones encontradas:", len(sections))
    print("Chunks generados:", len(chunks))

    for index, chunk in enumerate(chunks, start=1):
        title = chunk.metadata.get(
            "section_title",
            chunk.metadata.get("document_title", "Sin título"),
        )
        print(
            f"{index}. {title} | "
            f"parte {chunk.metadata['part']}/"
            f"{chunk.metadata['total_parts']} | "
            f"{chunk.metadata['token_count']} tokens"
        )

import pytest

from src.chunking import split_into_chunks, split_sections
from src.document_loader import load_documents


@pytest.fixture(scope="session")
def documents():
    return load_documents()


@pytest.fixture(scope="session")
def sections(documents):
    return split_sections(documents)


@pytest.fixture(scope="session")
def chunks(sections):
    return split_into_chunks(sections)

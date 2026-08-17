import pytest

from src import retrieval


def test_retrieve_chunks_rechaza_pregunta_vacia():
    with pytest.raises(ValueError, match="vacía"):
        retrieval.retrieve_chunks("   ")


def test_retrieve_chunks_rechaza_top_k_fuera_de_rango():
    with pytest.raises(ValueError, match="entre 2 y 5"):
        retrieval.retrieve_chunks("vacaciones", top_k=1)

    with pytest.raises(ValueError, match="entre 2 y 5"):
        retrieval.retrieve_chunks("vacaciones", top_k=6)


def test_retrieve_chunks_rechaza_indice_vacio(monkeypatch):
    class EmptyCollection:
        @staticmethod
        def count():
            return 0

    class EmptyVectorStore:
        _collection = EmptyCollection()

    monkeypatch.setattr(retrieval, "load_vector_store", lambda: EmptyVectorStore())

    with pytest.raises(ValueError, match="índice vectorial está vacío"):
        retrieval.retrieve_chunks("vacaciones")

import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

if not EMBEDDING_MODEL:
    raise ValueError("Falta definir EMBEDDING_MODEL en el archivo .env")


def get_embedding_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)
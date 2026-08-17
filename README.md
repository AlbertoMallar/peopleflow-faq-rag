# Proyecto Integrador Módulo 2 - FAQ RAG de PeopleFlow

Este repositorio implementa un pipeline RAG sobre una base de conocimiento de PeopleFlow para responder preguntas frecuentes de RR.HH. con trazabilidad hacia los chunks recuperados. El proyecto está pensado para ejecutarse tal como está desde un repositorio Git público, con configuración vía `.env`, persistencia local en Chroma y una documentación única que explica carga del corpus, chunking, embeddings, retrieval, generación de respuesta, evaluación y tests sin depender de materiales externos no documentados.

## Objetivo

Construir un sistema que:

- indexe `data/faq_document.txt`;
- genere embeddings con `text-embedding-3-small`;
- persista los chunks en Chroma local;
- recupere `TOP_K=3` chunks relevantes;
- responda con `gpt-4o-mini` usando únicamente evidencia recuperada;
- devuelva un JSON público con exactamente `user_question`, `system_answer` y `chunks_related`.

## Qué es RAG y por qué se usa acá

Esto utiliza RAG porque primero recupera chunks relevantes de la base de conocimiento y recién después utiliza esos chunks como contexto para generar la respuesta.

Beneficios en este proyecto:

- conocimiento actualizable sin reentrenar el modelo;
- grounding sobre la documentación real;
- trazabilidad y atribución mediante `chunks_related`.

## Documento fuente

- Archivo: `data/faq_document.txt`
- Formato: texto plano UTF-8
- Tamaño actual verificado: 3647 palabras
- Procesamiento actual: 100% del documento
- Resultado actual del corpus: 26 secciones/chunks válidos

No se realiza limpieza agresiva del texto porque el corpus ya viene estructurado con headers y párrafos semánticos que conviene preservar.

## Arquitectura

### Indexing pipeline

`documento -> load_documents() -> split_sections() -> split_into_chunks() -> validate_chunks() -> embeddings -> Chroma persistente`

### Query pipeline

`question -> query embedding -> vector retrieval -> build_context() -> LLM generation -> RAGResponse`

En la etapa de query, el embedding de la pregunta se genera internamente dentro de `Chroma.similarity_search_with_relevance_scores(...)` porque el vector store se carga con `embedding_function=get_embedding_model()`, que usa el mismo `EMBEDDING_MODEL` configurado en `.env`.

## Estructura del repositorio

```text
data/
  faq_document.txt
outputs/
  sample_queries.json
src/
  build_index.py
  chunking.py
  document_loader.py
  embeddings.py
  evaluate_retrieval.py
  evaluator.py
  index.py
  query.py
  retrieval.py
  schemas.py
  vector_store.py
storage/
  chroma/
tests/
README.md
requirements.txt
.env.example
.gitignore
```

## Requisitos previos

- Python `3.14.6`
- Una `OPENAI_API_KEY` válida en `.env`

## Instalación

### 1. Crear y activar el entorno virtual

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Configurar variables de entorno

Crear `.env` a partir de `.env.example` y completar la API key real.

`.env.example` incluye:

```dotenv
OPENAI_API_KEY=your-key-here
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

Variables usadas:

- `OPENAI_API_KEY`: credencial para embeddings, query generation y evaluator bonus.
- `EMBEDDING_MODEL`: debe permanecer en `text-embedding-3-small`.
- `LLM_MODEL`: debe permanecer en `gpt-4o-mini`.

`.env` no debe subirse a Git. El repositorio incluye `.gitignore` para evitarlo.

### 3. Instalar dependencias

```powershell
pip install -r requirements.txt
```

## Cómo construir el índice

Entry point esperado por la consigna:

```powershell
python -m src.build_index
```

`src/build_index.py` es un wrapper fino sobre `src/index.py` para no duplicar la lógica ya validada.

El índice en `storage/chroma/` es generado por este pipeline y no necesita venir versionado en el repositorio. Cualquier clon del proyecto puede reconstruirlo a partir de `data/faq_document.txt`, el código fuente y una `OPENAI_API_KEY` válida.

## Cómo hacer una consulta

```powershell
python -m src.query --question "¿Cómo solicito vacaciones?"
```

### Formato esperado del JSON público

```json
{
  "user_question": "¿Cómo solicito vacaciones?",
  "system_answer": "Respuesta grounded...",
  "chunks_related": [
    {
      "chunk_id": "abcd1234efgh5678",
      "section_title": "Vacaciones",
      "relevance_score": 0.91,
      "content": "..."
    }
  ]
}
```

El contrato público mantiene exactamente estas tres claves raíz:

- `user_question`
- `system_answer`
- `chunks_related`

No se agregan `scores`, `evaluator` ni otras claves al JSON público.

## Estrategia de chunking

La estrategia está implementada en `src/chunking.py` y sigue esta lógica:

1. primero se usa la estructura semántica del documento mediante headers Markdown `#` y `##`;
2. luego se mide la cantidad real de tokens con `tiktoken` usando el encoding del `EMBEDDING_MODEL`;
3. si una sección completa entra en `MAX_SECTION_TOKENS=300`, se conserva como una unidad semántica;
4. si una sección supera ese umbral, se subdivide con `RecursiveCharacterTextSplitter` token-aware en chunks objetivo de `200` tokens con overlap `30`;
5. el overlap se aplica únicamente dentro de la misma sección y nunca cruza títulos distintos.

### Parámetros actuales

- `MAX_SECTION_TOKENS = 300`
- `SPLIT_CHUNK_SIZE = 200`
- `CHUNK_OVERLAP = 30`

### Justificación técnica

- Antes de incorporar títulos, las 26 secciones del corpus actual midieron entre `115` y `256` tokens.
- Eso permitió elegir `MAX_SECTION_TOKENS=300` para no fragmentar innecesariamente unidades semánticas coherentes.
- Cuando una sección supere `300`, el objetivo pasa a ser `200` tokens para mantener granularidad útil en retrieval.
- El overlap `30` sólo existe al subdividir una sección grande.
- El overlap nunca cruza límites entre secciones.
- Esta estrategia evita el corte fijo arbitrario cada N tokens, reduce fragmentación y overlap innecesario y preserva mejor el contexto.

### Resultado actual verificado

- 100% del documento procesado
- 26 chunks distintos
- todos entre `127` y `259` tokens

Además, `src/chunking.py` conserva un bloque diagnóstico desactivado con `if False` para mostrar cómo se midieron los tokens por sección antes de elegir los parámetros.

## Embeddings

- Modelo: `text-embedding-3-small`
- Configuración: `.env`
- Implementación: `src/embeddings.py`
- Un embedding por chunk
- Persistencia: Chroma local en `storage/chroma`

La colección persistida actual contiene 26 documentos, consistente con los 26 chunks generados por el corpus actual.

## Vector store y búsqueda vectorial

- Vector store: `langchain_chroma.Chroma`
- Persistencia local: `storage/chroma`
- Método de búsqueda invocado: `similarity_search_with_relevance_scores()`
- `TOP_K`: `3`
- Métrica/espacio actual: `l2`

### Aclaración importante sobre la métrica real

La implementación deja explícito `COLLECTION_CONFIGURATION = {"hnsw": {"space": "l2"}}` en `src/vector_store.py`. Además, al inspeccionar la colección persistida actual, la configuración reporta `hnsw.space = l2`.

Esto significa que:

- Chroma indexa la colección con espacio `l2`;
- `similarity_search_with_relevance_scores()` hace la búsqueda sobre esa configuración;
- LangChain normaliza luego esa distancia para exponer `relevance_score` en rango `[0, 1]`.

No se asume coseno porque la configuración actual real no lo establece.

### Justificación de `TOP_K=3`

`TOP_K` se mantiene en `3` porque:

- cumple con la consigna de recuperar entre 2 y 5 chunks;
- reduce ruido respecto de valores mayores;
- coincide con `outputs/sample_queries.json`;
- es el valor usado por la evaluación reproducible de retrieval.

## Manejo de preguntas sin evidencia suficiente

El prompt RAG indica explícitamente que, si el contexto recuperado no alcanza para responder, el sistema debe devolver:

`No tengo información suficiente en la documentación disponible.`

Esto evita alucinaciones y mantiene el grounding sobre la documentación.

## Sample queries

`outputs/sample_queries.json` ya contiene 4 consultas reales y cada ejemplo conserva `3` elementos en `chunks_related`, coherentes con `TOP_K=3`.

No hace falta regenerarlo salvo que cambie el contrato público o el comportamiento validado del pipeline.

## Evaluación de retrieval

Script reproducible:

```powershell
python -m src.evaluate_retrieval
```

Métricas implementadas:

- `Precision@3` por consulta
- `Hit@3` por consulta
- promedio de `Precision@3`

Gold labels usados:

1. `¿Cómo solicito vacaciones?`
2. `¿Qué hago si pierdo el acceso al segundo factor de autenticación?`
3. `¿Cómo puedo importar empleados mediante un archivo CSV?`
4. `¿Puede un líder modificar el saldo de vacaciones de su equipo?`

El script usa exactamente esas cuatro preguntas. Si se ejecuta, consume sólo las llamadas de embedding necesarias para esas consultas y no usa el LLM generativo.

## Evaluador bonus

Archivo:

```text
src/evaluator.py
```

Características:

- completamente separado del JSON público del RAG;
- usa `ChatPromptTemplate`;
- usa `ChatOpenAI` con `LLM_MODEL` actual;
- fija `temperature=0`;
- usa `with_structured_output(...)`;
- valida nuevamente el resultado con `validate_evaluation()`.

Contrato del evaluator:

```json
{
  "score": 8,
  "reason": "Justificación en castellano con al menos 50 caracteres..."
}
```

## Tests

Ejecutar:

```powershell
python -m pytest
```

La suite cubre, sin usar OpenAI real:

- carga del corpus;
- cantidad de chunks;
- rango de tokens por chunk;
- conservación de metadata clave;
- determinismo de `chunk_id`;
- validación de pregunta vacía;
- validación de `TOP_K` entre 2 y 5;
- contrato exacto del JSON público;
- validación del evaluator;
- lógica principal de query con mocks;
- consistencia de `outputs/sample_queries.json`.

## Decisiones técnicas

- Se preservó el pipeline RAG existente en lugar de rediseñarlo.
- `src/build_index.py` se agregó como entry point fino para cumplir la consigna sin duplicar lógica.
- La métrica `l2` quedó explícita en la configuración del vector store para que la rúbrica pueda auditarla sin ambigüedad.
- Se mantuvo el prompt RAG ya validado; sólo se extrajo a una constante para mejorar modularidad sin alterar su contenido.
- Se agregaron validaciones de errores básicos que no cambian el comportamiento normal: pregunta vacía, `TOP_K` inválido, índice inexistente o vacío y evaluación inválida.

## Notas sobre artefactos

- `storage/chroma/` es regenerable mediante `python -m src.build_index`.
- El directorio está ignorado en Git para no versionar artefactos pesados o regenerables.
- `outputs/sample_queries.json` sí se conserva como evidencia de consultas reales.

## Reconstrucción mínima del proyecto

Secuencia recomendada desde un clon limpio:

1. crear y activar el entorno virtual;
2. configurar `.env`;
3. instalar `requirements.txt`;
4. ejecutar `python -m src.build_index`;
5. consultar con `python -m src.query --question "¿Cómo solicito vacaciones?"`.

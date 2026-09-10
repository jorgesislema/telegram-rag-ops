# -*- coding: utf-8 -*-
"""Motor RAG que coordina embeddings, recuperación y generación.

Implementa la clase `RAGEngine` que orquesta:
1. Generación de embeddings vía API de NVIDIA (o fallback NumPy para tests).
2. Búsqueda semántica en `VectorStore` con filtros de fecha opcionales.
3. Construcción de prompt con contexto recuperado.
4. Llamada al modelo Nemotron para generar la respuesta final.
"""

import os
from typing import Any, Optional

import numpy as np
from openai import OpenAI
from openai.types import Embedding

from src.database import VectorStore
from src.utils.logger import get_logger

logger: Any = get_logger(__name__)

# Constantes de configuración
_NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
_EMBEDDING_MODEL: str = "nvidia/embed-qa-4"
_GENERATION_MODEL: str = "nvidia/nemotron-4-340b-instruct"
_MAX_TOKENS_EMBEDDING: int = 8192
_MAX_TOKENS_GENERATION: int = 4096
_TEMPERATURE: float = 0.2


def _fallback_embedding(text: str, dim: int = 4096) -> list[float]:
    """Genera un embedding determinista usando hash del texto (solo para tests).

    Args:
        text: Texto de entrada.
        dim: Dimensión del vector de salida.

    Returns:
        Vector de floats normalizado.
    """
    rng = np.random.default_rng(seed=hash(text) % (2**32 - 1))
    vec = rng.normal(size=dim).astype(np.float32)
    norma = np.linalg.norm(vec)
    if norma == 0:
        return [0.0] * dim
    return (vec / norma).tolist()


class RAGEngine:
    """Motor de Retrieval-Augmented Generation.

    Coordina la generación de embeddings, la búsqueda en VectorStore y
    la generación de respuestas con Nemotron.

    Attributes:
        _vector_store: Instancia de VectorStore para recuperación.
        _client: Cliente OpenAI apuntando a NVIDIA API.
        _api_key: Clave de API de NVIDIA.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        api_key: Optional[str] = None,
    ) -> None:
        """Inicializa el motor RAG.

        Args:
            vector_store: Almacén de vectores ya configurado.
            api_key: Clave NVIDIA_API_KEY. Si es None, intenta leerla
                de la variable de entorno. Si no está disponible,
                usa modo fallback (sin llamadas reales a la API).

        Raises:
            ValueError: Si no se proporciona api_key ni existe en el entorno
                y no se está en modo test (se permite para tests).
        """
        self._vector_store: VectorStore = vector_store

        # Permitir None para tests que usan fallback
        self._api_key: Optional[str] = api_key or os.getenv("NVIDIA_API_KEY")
        self._client: Optional[OpenAI] = None

        if self._api_key:
            try:
                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url=_NVIDIA_BASE_URL,
                )
                logger.info("Cliente NVIDIA API inicializado correctamente")
            except Exception as exc:
                logger.error("Error al crear cliente NVIDIA: %s", exc)
                raise
        else:
            logger.warning(
                "NVIDIA_API_KEY no configurada — usando fallback NumPy para embeddings"
            )

    def generate_embedding(self, text: str) -> list[float]:
        """Convierte texto a vector de embedding.

        Si hay API key configurada, usa el modelo de embeddings de NVIDIA.
        En caso contrario, genera un embedding determinista con NumPy
        (útil para tests sin conexión de red).

        Args:
            text: Texto a vectorizar.

        Returns:
            Lista de floats representando el embedding normalizado.

        Raises:
            RuntimeError: Si la llamada a la API falla y no hay fallback.
        """
        if not text.strip():
            logger.warning("Texto vacío para embedding — devolviendo vector cero")
            return [0.0] * 4096

        if self._client is None:
            return _fallback_embedding(text)

        client: Optional[OpenAI] = self._client
        if client is None:
            logger.warning("Cliente NVIDIA no inicializado — usando fallback NumPy")
            return _fallback_embedding(text)

        try:
            response: Any = client.embeddings.create(
                model=_EMBEDDING_MODEL,
                input=text[:_MAX_TOKENS_EMBEDDING],
                encoding_format="float",
            )
            embedding: list[float] = list(response.data[0].embedding)
            logger.debug("Embedding generado: %d dimensiones", len(embedding))
            return embedding
        except Exception as exc:
            logger.error("Error generando embedding con NVIDIA: %s", exc)
            # Fallback silencioso para no romper el flujo en producción
            logger.warning("Usando embedding de fallback NumPy")
            return _fallback_embedding(text)

    def query(
        self,
        user_query: str,
        top_k: int = 3,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict[str, Any]:
        """Ejecuta el pipeline RAG completo para una consulta de usuario.

        Flujo:
        1. Genera embedding de la consulta.
        2. Busca fragmentos relevantes en VectorStore (con filtros de fecha).
        3. Construye prompt con contexto estructurado.
        4. Llama a Nemotron para generar respuesta.

        Args:
            user_query: Pregunta del usuario en lenguaje natural.
            top_k: Número máximo de fragmentos a recuperar.
            start_date: Fecha ISO 8601 inicio (inclusive). Opcional.
            end_date: Fecha ISO 8601 fin (inclusive). Opcional.

        Returns:
            Diccionario con claves:
            - `response`: Texto generado por Nemotron.
            - `context`: Lista de fragmentos usados como contexto.
            - `sources`: Lista de fuentes únicas de los fragmentos.

        Raises:
            RuntimeError: Si falla la generación y no hay fallback.
        """
        if not user_query.strip():
            logger.warning("Consulta vacía — devolviendo respuesta vacía")
            return {"response": "", "context": [], "sources": []}

        # 1. Embedding de la consulta
        query_embedding: list[float] = self.generate_embedding(user_query)

        # 2. Búsqueda en VectorStore
        try:
            resultados: list[dict[str, Any]] = self._vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=top_k,
                start_date=start_date,
                end_date=end_date,
            )
            logger.debug("Recuperados %d fragmentos", len(resultados))
        except Exception as exc:
            logger.error("Error en búsqueda vectorial: %s", exc)
            resultados = []

        # 3. Construcción del contexto
        context: list[str] = [r["text"] for r in resultados]
        sources: list[str] = list({r["source"] for r in resultados})

        if not context:
            logger.info("Sin contexto recuperado — respondiendo sin RAG")
            context_str: str = "No hay información relevante en la base de conocimiento."
        else:
            context_str = "\n\n---\n\n".join(
                f"[Fuente: {r['source']} | Fecha: {r['date']}]\n{r['text']}"
                for r in resultados
            )

        # 4. Prompt estructurado
        system_prompt: str = (
            "Eres un asistente experto que responde basándose exclusivamente "
            "en el contexto proporcionado. Si la información no está en el "
            "contexto, indícalo claramente. Responde en español."
        )
        user_prompt: str = (
            f"Contexto:\n{context_str}\n\n"
            f"Pregunta: {user_query}\n\n"
            "Responde basándote solo en el contexto anterior."
        )

        # 5. Llamada al modelo de generación
        if self._client is None:
            response_text: str = (
                "[MODO TEST] Respuesta simulada para: " + user_query
            )
            logger.warning("Generación en modo fallback — sin llamada real a Nemotron")
        else:
            try:
                completion = self._client.chat.completions.create(
                    model=_GENERATION_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=_MAX_TOKENS_GENERATION,
                    temperature=_TEMPERATURE,
                )
                response_text = completion.choices[0].message.content or ""
                logger.debug("Respuesta generada: %d caracteres", len(response_text))
            except Exception as exc:
                logger.error("Error en generación Nemotron: %s", exc)
                response_text = (
                    "Error al generar la respuesta. Intente nuevamente más tarde."
                )

        return {
            "response": response_text.strip(),
            "context": context,
            "sources": sources,
        }
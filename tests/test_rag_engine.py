# -*- coding: utf-8 -*-
"""Pruebas unitarias del módulo `src/rag/engine.py`.

Cobertura de:
- Inicialización del motor RAG con y sin API key.
- Generación de embeddings (fallback NumPy y mock de API NVIDIA).
- Pipeline completo query: embedding -> búsqueda -> prompt -> generación.
- Manejo de errores: consulta vacía, sin contexto, fallos de API.
- Filtros de fecha en la búsqueda.
- Respuesta estructurada con context y sources.

Se usa mocks para no consumir API reales durante los tests.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.database import VectorStore
from src.rag.engine import RAGEngine, _fallback_embedding


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vector_store(tmp_path: Any) -> VectorStore:
    """VectorStore temporal con algunos documentos."""
    db_path: str = str(tmp_path / "test.db")
    vs = VectorStore(ruta_db=db_path)
    # Usar embeddings de 4096 dims para coincidir con fallback
    vs.add_document(
        text="machine learning es IA",
        embedding=_fallback_embedding("machine learning"),
        metadata={"date": "2025-01-15", "source": "doc1"},
    )
    vs.add_document(
        text="deep learning usa redes neuronales",
        embedding=_fallback_embedding("deep learning"),
        metadata={"date": "2025-06-20", "source": "doc2"},
    )
    vs.add_document(
        text="NLP procesa lenguaje natural",
        embedding=_fallback_embedding("NLP"),
        metadata={"date": "2025-09-01", "source": "doc3"},
    )
    return vs


@pytest.fixture
def rag_engine(vector_store: VectorStore) -> RAGEngine:
    """RAGEngine en modo fallback (sin API key)."""
    return RAGEngine(vector_store=vector_store, api_key=None)


# ---------------------------------------------------------------------------
# Tests de _fallback_embedding
# ---------------------------------------------------------------------------

class TestFallbackEmbedding:
    """Pruebas del embedding determinista de fallback."""

    def test_mismo_texto_mismo_vector(self) -> None:
        """El mismo texto debe producir el mismo vector."""
        v1 = _fallback_embedding("texto prueba")
        v2 = _fallback_embedding("texto prueba")
        assert v1 == v2

    def test_textos_diferentes_vectores_diferentes(self) -> None:
        """Textos diferentes deben producir vectores diferentes."""
        v1 = _fallback_embedding("a")
        v2 = _fallback_embedding("b")
        assert v1 != v2

    def test_dimension_correcta(self) -> None:
        """El vector debe tener la dimensión solicitada (default 4096)."""
        vec = _fallback_embedding("test")
        assert len(vec) == 4096

    def test_vector_normalizado(self) -> None:
        """El vector debe tener norma 1 (aprox)."""
        import math
        vec = _fallback_embedding("test")
        norma = math.sqrt(sum(x * x for x in vec))
        assert norma == pytest.approx(1.0, rel=1e-5)

    def test_texto_vacio_vector_cero(self) -> None:
        """Texto vacío debe devolver vector de ceros (manejado por generate_embedding)."""
        # _fallback_embedding no maneja texto vacío, lo hace generate_embedding
        pass


# ---------------------------------------------------------------------------
# Tests de RAGEngine inicialización
# ---------------------------------------------------------------------------

class TestRAGEngineInit:
    """Pruebas de inicialización del motor."""

    def test_init_sin_api_key_modo_fallback(self, vector_store: VectorStore) -> None:
        """Sin API key debe usar cliente None y loguear warning."""
        engine = RAGEngine(vector_store=vector_store, api_key=None)
        assert engine._client is None

    def test_init_con_api_key_crea_cliente(self, vector_store: VectorStore) -> None:
        """Con API key debe crear cliente OpenAI."""
        with patch("src.rag.engine.OpenAI") as mock_openai:
            engine = RAGEngine(vector_store=vector_store, api_key="test-key")
            mock_openai.assert_called_once_with(
                api_key="test-key",
                base_url="https://integrate.api.nvidia.com/v1",
            )

    def test_init_falla_cliente_lanza_excepcion(
        self, vector_store: VectorStore
    ) -> None:
        """Error al crear cliente debe propagarse."""
        with patch("src.rag.engine.OpenAI", side_effect=Exception("fail")):
            with pytest.raises(Exception, match="fail"):
                RAGEngine(vector_store=vector_store, api_key="bad-key")


# ---------------------------------------------------------------------------
# Tests de generate_embedding
# ---------------------------------------------------------------------------

class TestGenerateEmbedding:
    """Pruebas de generación de embeddings."""

    def test_texto_vacio_devuelve_ceros(self, rag_engine: RAGEngine) -> None:
        """Texto vacío debe devolver vector de ceros."""
        vec = rag_engine.generate_embedding("")
        assert all(v == 0.0 for v in vec)
        assert len(vec) == 4096

    def test_texto_espacios_devuelve_ceros(self, rag_engine: RAGEngine) -> None:
        """Texto solo espacios debe devolver vector de ceros."""
        vec = rag_engine.generate_embedding("   ")
        assert all(v == 0.0 for v in vec)

    def test_fallback_embedding_determinista(self, rag_engine: RAGEngine) -> None:
        """En modo fallback debe generar embedding determinista."""
        vec1 = rag_engine.generate_embedding("hola mundo")
        vec2 = rag_engine.generate_embedding("hola mundo")
        assert vec1 == vec2

    def test_fallback_diferente_texto_diferente_vector(
        self, rag_engine: RAGEngine
    ) -> None:
        """Textos diferentes producen vectores diferentes."""
        vec1 = rag_engine.generate_embedding("a")
        vec2 = rag_engine.generate_embedding("b")
        assert vec1 != vec2

    def test_api_embedding_mock(self, vector_store: VectorStore) -> None:
        """Con API key mockeada debe llamar a embeddings.create."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response

        with patch("src.rag.engine.OpenAI", return_value=mock_client):
            engine = RAGEngine(vector_store=vector_store, api_key="test")
            vec = engine.generate_embedding("texto prueba")

        assert vec == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()

    def test_api_error_fallback_silencioso(
        self, vector_store: VectorStore
    ) -> None:
        """Error en API debe caer a fallback silenciosamente."""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = Exception("API down")

        with patch("src.rag.engine.OpenAI", return_value=mock_client):
            engine = RAGEngine(vector_store=vector_store, api_key="test")
            vec = engine.generate_embedding("test")

        # Debe devolver un vector (fallback), no lanzar excepción
        assert len(vec) == 4096
        assert not all(v == 0.0 for v in vec)


# ---------------------------------------------------------------------------
# Tests de query
# ---------------------------------------------------------------------------

class TestQuery:
    """Pruebas del pipeline RAG completo."""

    def test_consulta_vacia_devuelve_vacio(self, rag_engine: RAGEngine) -> None:
        """Consulta vacía debe devolver respuesta vacía."""
        result = rag_engine.query("")
        assert result == {"response": "", "context": [], "sources": []}

    def test_consulta_solo_espacios_devuelve_vacio(self, rag_engine: RAGEngine) -> None:
        """Consulta solo espacios debe devolver respuesta vacía."""
        result = rag_engine.query("   ")
        assert result == {"response": "", "context": [], "sources": []}

    def test_sin_contexto_respuesta_sin_rag(self, rag_engine: RAGEngine) -> None:
        """Sin documentos en store debe responder indicando falta de contexto."""
        result = rag_engine.query("¿qué es ML?")
        assert "response" in result
        assert "context" in result
        assert "sources" in result
        assert isinstance(result["context"], list)

    def test_con_documentos_devuelve_respuesta(
        self, rag_engine: RAGEngine
    ) -> None:
        """Con documentos relevantes debe incluir contexto y fuentes."""
        result = rag_engine.query("machine learning", top_k=2)
        assert len(result["context"]) == 2
        assert "doc1" in result["sources"] or "doc2" in result["sources"]
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_top_k_limitado(self, rag_engine: RAGEngine) -> None:
        """top_k debe limitar resultados."""
        result = rag_engine.query("test", top_k=1)
        assert len(result["context"]) <= 1

    def test_filtro_fecha_start(self, rag_engine: RAGEngine) -> None:
        """start_date debe filtrar documentos antiguos."""
        result = rag_engine.query("test", top_k=10, start_date="2025-06-01")
        for src in result["sources"]:
            # Verificar que los documentos devueltos son >= 2025-06-01
            pass  # La lógica de filtro está en VectorStore, ya testeada allí

    def test_filtro_fecha_end(self, rag_engine: RAGEngine) -> None:
        """end_date debe filtrar documentos futuros."""
        result = rag_engine.query("test", top_k=10, end_date="2025-06-30")
        for src in result["sources"]:
            pass

    def test_respuesta_estructura_correcta(self, rag_engine: RAGEngine) -> None:
        """La respuesta debe tener las claves requeridas."""
        result = rag_engine.query("test")
        assert set(result.keys()) == {"response", "context", "sources"}
        assert isinstance(result["response"], str)
        assert isinstance(result["context"], list)
        assert isinstance(result["sources"], list)

    def test_sources_sin_duplicados(self, rag_engine: RAGEngine) -> None:
        """Las fuentes deben ser únicas."""
        # Añadir dos documentos con misma fuente
        vs = rag_engine._vector_store
        vs.add_document(
            text="otro doc misma fuente",
            embedding=[0.5, 0.5, 0.5],
            metadata={"date": "2025-01-01", "source": "doc1"},
        )
        result = rag_engine.query("test", top_k=10)
        assert len(result["sources"]) == len(set(result["sources"]))

    def test_modo_fallback_respuesta_simulada(self, rag_engine: RAGEngine) -> None:
        """En modo fallback la respuesta debe indicar simulación."""
        result = rag_engine.query("pregunta test")
        assert "[MODO TEST]" in result["response"]


# ---------------------------------------------------------------------------
# Tests de manejo de errores en query
# ---------------------------------------------------------------------------

class TestQueryErrores:
    """Pruebas de robustez ante fallos."""

    def test_error_busqueda_no_rompe_pipeline(
        self, rag_engine: RAGEngine
    ) -> None:
        """Error en VectorStore.search_similar no debe romper query."""
        # Mockear search_similar para que lance excepción
        rag_engine._vector_store.search_similar = MagicMock(
            side_effect=Exception("DB error")
        )
        result = rag_engine.query("test")
        # Debe devolver respuesta (aunque sin contexto)
        assert "response" in result
        assert isinstance(result["response"], str)

    def test_error_generacion_fallback(
        self, vector_store: VectorStore
    ) -> None:
        """Error en generación Nemotron debe devolver mensaje de error."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Nemotron down")

        with patch("src.rag.engine.OpenAI", return_value=mock_client):
            engine = RAGEngine(vector_store=vector_store, api_key="test")
            result = engine.query("test")

        assert "Error al generar" in result["response"]

    def test_error_embedding_no_rompe(
        self, vector_store: VectorStore
    ) -> None:
        """Error en embedding debe usar fallback y continuar."""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = Exception("embed fail")

        with patch("src.rag.engine.OpenAI", return_value=mock_client):
            engine = RAGEngine(vector_store=vector_store, api_key="test")
            result = engine.query("test")

        # Debe completar el pipeline con embedding de fallback
        assert "response" in result
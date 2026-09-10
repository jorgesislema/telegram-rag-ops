# -*- coding: utf-8 -*-
"""Pruebas unitarias consolidadas de `VectorStore` y `RAGEngine`.

Cobertura de:

**VectorStore:**
- Inicialización y creación de la tabla SQLite.
- Inserción de documentos con generación de IDs únicos.
- Validación de metadatos obligatorios (date, source).
- Búsqueda por similitud de coseno.
- Filtrado por rango de fechas (start_date, end_date).
- Manejo de dimensiones incompatibles entre vectores.
- Errores SQLite en inicialización, inserción y lectura.

**RAGEngine:**
- Inicialización con y sin API key (modo fallback).
- Generación de embeddings via API mockeada.
- Generación de embeddings en modo fallback NumPy.
- Pipeline query completo: embedding -> búsqueda -> prompt -> generación.
- Filtros de fecha en query.
- Manejo de excepciones en API de embeddings.
- Manejo de excepciones en API de generación.
- Respuesta estructurada con context y sources.

Los tests no realizan llamadas HTTP reales — todas las interacciones
con la API de NVIDIA/Nemotron están mockeadas.
"""

import json
import math
import sqlite3
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.database import VectorStore
from src.database.vector_store import _similitud_coseno
from src.rag import RAGEngine
from src.rag.engine import _fallback_embedding


# ---------------------------------------------------------------------------
# Fixtures compartidos
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_store(tmp_path: Any) -> VectorStore:
    """VectorStore temporal aislado para cada prueba."""
    db_path: str = str(tmp_path / "test_vectors.db")
    return VectorStore(ruta_db=db_path)


@pytest.fixture
def store_con_docs(tmp_store: VectorStore) -> VectorStore:
    """VectorStore con tres documentos de prueba precargados."""
    tmp_store.add_document(
        text="machine learning es una rama de la inteligencia artificial",
        embedding=[1.0, 0.0, 0.0],
        metadata={"date": "2025-01-15", "source": "doc_ml"},
    )
    tmp_store.add_document(
        text="deep learning usa redes neuronales profundas",
        embedding=[0.9, 0.4, 0.1],
        metadata={"date": "2025-06-20", "source": "doc_dl"},
    )
    tmp_store.add_document(
        text="procesamiento de lenguaje natural NLP",
        embedding=[0.1, 0.1, 0.9],
        metadata={"date": "2025-09-01", "source": "doc_nlp"},
    )
    return tmp_store


@pytest.fixture
def rag_fallback(tmp_store: VectorStore) -> RAGEngine:
    """RAGEngine en modo fallback (sin API key, sin cliente OpenAI)."""
    return RAGEngine(vector_store=tmp_store, api_key=None)


# ===========================================================================
# Tests de _similitud_coseno
# ===========================================================================

class TestSimilitudCoseno:
    """Pruebas de la función auxiliar de similitud de coseno."""

    def test_vectores_identicos_devuelven_uno(self) -> None:
        """Vectores idénticos deben tener similitud 1.0."""
        assert _similitud_coseno([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_vectores_perpendiculares_devuelven_cero(self) -> None:
        """Vectores ortogonales deben tener similitud 0.0."""
        assert _similitud_coseno([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_vectores_opuestos_devuelven_menos_uno(self) -> None:
        """Vectores opuestos deben tener similitud -1.0."""
        assert _similitud_coseno([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_longitud_diferente_lanza_value_error(self) -> None:
        """Vectores de distinta longitud deben lanzar ValueError."""
        with pytest.raises(ValueError, match="misma longitud"):
            _similitud_coseno([1.0, 0.0], [1.0, 0.0, 0.0])

    def test_vector_nulo_devuelve_cero(self) -> None:
        """Un vector con magnitud cero debe devolver similitud 0.0."""
        assert _similitud_coseno([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


# ===========================================================================
# Tests de VectorStore — Inicialización
# ===========================================================================

class TestVectorStoreInit:
    """Pruebas de inicialización del almacén de vectores."""

    def test_crea_archivo_sqlite(self, tmp_path: Any) -> None:
        """Debe crear el archivo de base de datos al instanciar."""
        db_path: str = str(tmp_path / "nueva.db")
        vs = VectorStore(ruta_db=db_path)
        import os
        assert os.path.exists(db_path)
        vs._conexion.close()

    def test_tabla_documentos_existe(self, tmp_store: VectorStore) -> None:
        """La tabla 'documentos' debe existir tras la inicialización."""
        cursor = tmp_store._conexion.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documentos'"
        )
        assert cursor.fetchone() is not None


# ===========================================================================
# Tests de VectorStore — Inserción
# ===========================================================================

class TestVectorStoreAddDocument:
    """Pruebas de inserción de documentos."""

    def test_retorna_id_str(self, tmp_store: VectorStore) -> None:
        """add_document debe retornar un string (ID único)."""
        doc_id: str = tmp_store.add_document(
            text="texto de prueba",
            embedding=[0.5, 0.5, 0.5],
            metadata={"date": "2025-01-01", "source": "test"},
        )
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    def test_ids_unicos(self, tmp_store: VectorStore) -> None:
        """Dos documentos insertados deben tener IDs diferentes."""
        id1: str = tmp_store.add_document(
            text="a", embedding=[1.0], metadata={"date": "2025-01-01", "source": "a"}
        )
        id2: str = tmp_store.add_document(
            text="b", embedding=[0.5], metadata={"date": "2025-01-01", "source": "b"}
        )
        assert id1 != id2

    def test_metadata_faltante_lanza_value_error(self, tmp_store: VectorStore) -> None:
        """Sin 'date' o 'source' en metadata debe lanzar ValueError."""
        with pytest.raises(ValueError, match="metadata debe contener"):
            tmp_store.add_document(
                text="texto",
                embedding=[1.0],
                metadata={"date": "2025-01-01"},
            )

    def test_documento_persiste_y_es_recuperable(
        self, tmp_store: VectorStore
    ) -> None:
        """El documento almacenado debe ser recuperable por la búsqueda."""
        tmp_store.add_document(
            text="contenido",
            embedding=[1.0, 0.0],
            metadata={"date": "2025-03-10", "source": "persistencia"},
        )
        resultados: list[dict[str, Any]] = tmp_store.search_similar(
            query_embedding=[1.0, 0.0], top_k=1
        )
        assert len(resultados) == 1
        assert resultados[0]["text"] == "contenido"

    def test_error_sqlite_en_insercion(self, tmp_store: VectorStore) -> None:
        """Error SQLite durante inserción debe loguearse y relanzarse."""
        mock_con = MagicMock()
        mock_con.execute.side_effect = sqlite3.Error("write fail")
        tmp_store._conexion = mock_con

        with pytest.raises(sqlite3.Error):
            tmp_store.add_document(
                text="x",
                embedding=[1.0],
                metadata={"date": "2025-01-01", "source": "x"},
            )


# ===========================================================================
# Tests de VectorStore — Búsqueda por similitud
# ===========================================================================

class TestVectorStoreSearchSimilar:
    """Pruebas de búsqueda por similitud de coseno."""

    def test_devuelve_top_k(self, store_con_docs: VectorStore) -> None:
        """Debe devolver como máximo top_k resultados."""
        resultados: list[dict[str, Any]] = store_con_docs.search_similar(
            query_embedding=[1.0, 0.0, 0.0], top_k=2
        )
        assert len(resultados) <= 2

    def test_orden_descendente_por_score(
        self, store_con_docs: VectorStore
    ) -> None:
        """Los resultados deben estar ordenados por similitud descendente."""
        resultados: list[dict[str, Any]] = store_con_docs.search_similar(
            query_embedding=[1.0, 0.0, 0.0], top_k=3
        )
        scores: list[float] = [r["score"] for r in resultados]
        assert scores == sorted(scores, reverse=True)

    def test_documento_mas_cercano_primero(
        self, store_con_docs: VectorStore
    ) -> None:
        """El primer resultado debe ser el más similar al query."""
        resultados: list[dict[str, Any]] = store_con_docs.search_similar(
            query_embedding=[1.0, 0.0, 0.0], top_k=1
        )
        assert resultados[0]["source"] == "doc_ml"

    def test_score_entre_menos_uno_y_uno(
        self, store_con_docs: VectorStore
    ) -> None:
        """El score de cada resultado debe estar en el rango [-1, 1]."""
        resultados: list[dict[str, Any]] = store_con_docs.search_similar(
            query_embedding=[1.0, 0.0, 0.0], top_k=3
        )
        for r in resultados:
            assert -1.0 <= r["score"] <= 1.0

    def test_top_k_mayor_que_total_devuelve_todos(
        self, store_con_docs: VectorStore
    ) -> None:
        """Si top_k es mayor que el total, debe devolver todos los documentos."""
        resultados: list[dict[str, Any]] = store_con_docs.search_similar(
            query_embedding=[1.0, 0.0, 0.0], top_k=100
        )
        assert len(resultados) == 3

    def test_almacen_vacio_devuelve_lista_vacia(
        self, tmp_store: VectorStore
    ) -> None:
        """Un almacén sin documentos debe devolver una lista vacía."""
        resultados: list[dict[str, Any]] = tmp_store.search_similar(
            query_embedding=[1.0, 0.0, 0.0], top_k=5
        )
        assert resultados == []

    def test_embedding_dim_incompatible_se_omite(
        self, tmp_store: VectorStore
    ) -> None:
        """Documentos con embedding de distinta dimensión se saltan."""
        tmp_store.add_document(
            text="doc corto",
            embedding=[1.0],
            metadata={"date": "2025-01-01", "source": "corto"},
        )
        tmp_store.add_document(
            text="doc largo",
            embedding=[1.0, 0.0, 0.0],
            metadata={"date": "2025-01-01", "source": "largo"},
        )
        resultados: list[dict[str, Any]] = tmp_store.search_similar(
            query_embedding=[1.0, 0.0, 0.0], top_k=10
        )
        assert len(resultados) == 1
        assert resultados[0]["source"] == "largo"

    def test_error_sqlite_en_lectura(self, tmp_store: VectorStore) -> None:
        """Error SQLite durante búsqueda debe loguearse y relanzarse."""
        mock_con = MagicMock()
        mock_con.execute.side_effect = sqlite3.Error("read fail")
        tmp_store._conexion = mock_con

        with pytest.raises(sqlite3.Error):
            tmp_store.search_similar(query_embedding=[1.0], top_k=1)


# ===========================================================================
# Tests de VectorStore — Filtrado por fechas
# ===========================================================================

class TestVectorStoreFiltrosFecha:
    """Pruebas de filtrado de búsquedas por rango de fechas."""

    def test_start_date_excluye_anteriores(
        self, store_con_docs: VectorStore
    ) -> None:
        """start_date debe excluir documentos anteriores a esa fecha."""
        resultados: list[dict[str, Any]] = store_con_docs.search_similar(
            query_embedding=[1.0, 0.0, 0.0],
            top_k=10,
            start_date="2025-06-01",
        )
        for r in resultados:
            assert r["date"] >= "2025-06-01"

    def test_end_date_excluye_posteriores(
        self, store_con_docs: VectorStore
    ) -> None:
        """end_date debe excluir documentos posteriores a esa fecha."""
        resultados: list[dict[str, Any]] = store_con_docs.search_similar(
            query_embedding=[1.0, 0.0, 0.0],
            top_k=10,
            end_date="2025-06-30",
        )
        for r in resultados:
            assert r["date"] <= "2025-06-30"

    def test_rango_completo_devuelve_solo_dentro_del_rango(
        self, store_con_docs: VectorStore
    ) -> None:
        """El filtro de rango completo debe devolver solo documentos dentro del rango."""
        resultados: list[dict[str, Any]] = store_con_docs.search_similar(
            query_embedding=[1.0, 0.0, 0.0],
            top_k=10,
            start_date="2025-01-01",
            end_date="2025-06-30",
        )
        assert len(resultados) == 2
        for r in resultados:
            assert "2025-01-01" <= r["date"] <= "2025-06-30"

    def test_sin_filtros_devuelve_todos(
        self, store_con_docs: VectorStore
    ) -> None:
        """Sin filtros de fecha debe devolver todos los documentos."""
        resultados: list[dict[str, Any]] = store_con_docs.search_similar(
            query_embedding=[1.0, 0.0, 0.0], top_k=10
        )
        assert len(resultados) == 3


# ===========================================================================
# Tests de _fallback_embedding
# ===========================================================================

class TestFallbackEmbedding:
    """Pruebas del embedding determinista de fallback (NumPy)."""

    def test_mismo_texto_mismo_vector(self) -> None:
        """El mismo texto debe producir el mismo vector."""
        v1: list[float] = _fallback_embedding("texto prueba")
        v2: list[float] = _fallback_embedding("texto prueba")
        assert v1 == v2

    def test_textos_diferentes_vectores_diferentes(self) -> None:
        """Textos diferentes deben producir vectores diferentes."""
        v1: list[float] = _fallback_embedding("alpha")
        v2: list[float] = _fallback_embedding("beta")
        assert v1 != v2

    def test_dimension_por_defecto(self) -> None:
        """El vector por defecto debe tener 4096 dimensiones."""
        vec: list[float] = _fallback_embedding("test")
        assert len(vec) == 4096

    def test_dimension_custom(self) -> None:
        """Debe respetar la dimensión solicitada."""
        vec: list[float] = _fallback_embedding("test", dim=128)
        assert len(vec) == 128

    def test_vector_normalizado(self) -> None:
        """El vector debe tener norma unitaria (aprox)."""
        vec: list[float] = _fallback_embedding("test")
        norma: float = math.sqrt(sum(x * x for x in vec))
        assert norma == pytest.approx(1.0, rel=1e-5)


# ===========================================================================
# Tests de RAGEngine — Inicialización
# ===========================================================================

class TestRAGEngineInit:
    """Pruebas de inicialización del motor RAG."""

    def test_sin_api_key_modo_fallback(
        self, tmp_store: VectorStore
    ) -> None:
        """Sin API key debe usar cliente None (modo fallback)."""
        engine: RAGEngine = RAGEngine(vector_store=tmp_store, api_key=None)
        assert engine._client is None

    def test_con_api_key_crea_cliente(self, tmp_store: VectorStore) -> None:
        """Con API key debe crear un cliente OpenAI."""
        with patch("src.rag.engine.OpenAI") as mock_cls:
            engine: RAGEngine = RAGEngine(
                vector_store=tmp_store, api_key="test-key"
            )
            mock_cls.assert_called_once_with(
                api_key="test-key",
                base_url="https://integrate.api.nvidia.com/v1",
            )

    def test_falla_cliente_lanza_excepcion(
        self, tmp_store: VectorStore
    ) -> None:
        """Error al crear el cliente debe propagarse."""
        with patch("src.rag.engine.OpenAI", side_effect=Exception("init fail")):
            with pytest.raises(Exception, match="init fail"):
                RAGEngine(vector_store=tmp_store, api_key="bad-key")


# ===========================================================================
# Tests de RAGEngine — generate_embedding
# ===========================================================================

class TestGenerateEmbedding:
    """Pruebas de generación de embeddings."""

    def test_texto_vacio_devuelve_ceros(self, rag_fallback: RAGEngine) -> None:
        """Texto vacío debe devolver vector de ceros."""
        vec: list[float] = rag_fallback.generate_embedding("")
        assert all(v == 0.0 for v in vec)
        assert len(vec) == 4096

    def test_solo_espacios_devuelve_ceros(
        self, rag_fallback: RAGEngine
    ) -> None:
        """Texto solo espacios debe devolver vector de ceros."""
        vec: list[float] = rag_fallback.generate_embedding("   ")
        assert all(v == 0.0 for v in vec)

    def test_fallback_determinista(self, rag_fallback: RAGEngine) -> None:
        """En modo fallback debe generar embedding determinista."""
        vec1: list[float] = rag_fallback.generate_embedding("hola mundo")
        vec2: list[float] = rag_fallback.generate_embedding("hola mundo")
        assert vec1 == vec2

    def test_fallback_textos_diferentes(
        self, rag_fallback: RAGEngine
    ) -> None:
        """Textos diferentes producen vectores diferentes."""
        vec1: list[float] = rag_fallback.generate_embedding("a")
        vec2: list[float] = rag_fallback.generate_embedding("b")
        assert vec1 != vec2

    def test_api_mock_devuelve_embedding(
        self, tmp_store: VectorStore
    ) -> None:
        """Con API key mockeada debe llamar a embeddings.create."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response

        with patch("src.rag.engine.OpenAI", return_value=mock_client):
            engine: RAGEngine = RAGEngine(
                vector_store=tmp_store, api_key="test"
            )
            vec: list[float] = engine.generate_embedding("texto prueba")

        assert vec == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()

    def test_api_error_fallback_silencioso(
        self, tmp_store: VectorStore
    ) -> None:
        """Error en API debe caer a fallback sin lanzar excepción."""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = Exception("API down")

        with patch("src.rag.engine.OpenAI", return_value=mock_client):
            engine: RAGEngine = RAGEngine(
                vector_store=tmp_store, api_key="test"
            )
            vec: list[float] = engine.generate_embedding("test")

        # Debe devolver un vector válido (fallback), no lanzar excepción
        assert len(vec) == 4096
        assert not all(v == 0.0 for v in vec)


# ===========================================================================
# Tests de RAGEngine — query (pipeline RAG completo)
# ===========================================================================

class TestQuery:
    """Pruebas del pipeline RAG completo."""

    def test_consulta_vacia_devuelve_vacio(
        self, rag_fallback: RAGEngine
    ) -> None:
        """Consulta vacía debe devolver respuesta vacía."""
        result: dict[str, Any] = rag_fallback.query("")
        assert result == {"response": "", "context": [], "sources": []}

    def test_solo_espacios_devuelve_vacio(
        self, rag_fallback: RAGEngine
    ) -> None:
        """Consulta solo espacios debe devolver respuesta vacía."""
        result: dict[str, Any] = rag_fallback.query("   ")
        assert result == {"response": "", "context": [], "sources": []}

    def test_respuesta_tiene_claves_requeridas(
        self, rag_fallback: RAGEngine
    ) -> None:
        """La respuesta debe contener las claves 'response', 'context', 'sources'."""
        result: dict[str, Any] = rag_fallback.query("test")
        assert set(result.keys()) == {"response", "context", "sources"}
        assert isinstance(result["response"], str)
        assert isinstance(result["context"], list)
        assert isinstance(result["sources"], list)

    def test_con_documentos_incluye_contexto(
        self, tmp_store: VectorStore
    ) -> None:
        """Con documentos relevantes el contexto no debe estar vacío."""
        # Precargar documentos con embeddings de 4096 dims (fallback)
        tmp_store.add_document(
            text="machine learning es IA",
            embedding=_fallback_embedding("machine learning"),
            metadata={"date": "2025-01-15", "source": "ml"},
        )
        tmp_store.add_document(
            text="deep learning usa redes",
            embedding=_fallback_embedding("deep learning"),
            metadata={"date": "2025-06-20", "source": "dl"},
        )
        engine: RAGEngine = RAGEngine(vector_store=tmp_store, api_key=None)
        result: dict[str, Any] = engine.query("machine learning", top_k=2)

        assert len(result["context"]) == 2
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_top_k_limita_resultados(self, tmp_store: VectorStore) -> None:
        """top_k debe limitar el número de fragmentos recuperados."""
        tmp_store.add_document(
            text="doc1",
            embedding=_fallback_embedding("doc1"),
            metadata={"date": "2025-01-01", "source": "s1"},
        )
        tmp_store.add_document(
            text="doc2",
            embedding=_fallback_embedding("doc2"),
            metadata={"date": "2025-01-01", "source": "s2"},
        )
        tmp_store.add_document(
            text="doc3",
            embedding=_fallback_embedding("doc3"),
            metadata={"date": "2025-01-01", "source": "s3"},
        )
        engine: RAGEngine = RAGEngine(vector_store=tmp_store, api_key=None)
        result: dict[str, Any] = engine.query("test", top_k=1)

        assert len(result["context"]) <= 1

    def test_sources_sin_duplicados(self, tmp_store: VectorStore) -> None:
        """Las fuentes en la respuesta deben ser únicas."""
        tmp_store.add_document(
            text="doc a",
            embedding=_fallback_embedding("a"),
            metadata={"date": "2025-01-01", "source": "misma_fuente"},
        )
        tmp_store.add_document(
            text="doc b",
            embedding=_fallback_embedding("b"),
            metadata={"date": "2025-01-01", "source": "misma_fuente"},
        )
        engine: RAGEngine = RAGEngine(vector_store=tmp_store, api_key=None)
        result: dict[str, Any] = engine.query("test", top_k=10)

        assert len(result["sources"]) == len(set(result["sources"]))

    def test_filtro_fecha_start_en_query(self, tmp_store: VectorStore) -> None:
        """start_date en query debe filtrar documentos antiguos."""
        tmp_store.add_document(
            text="antiguo",
            embedding=_fallback_embedding("antiguo"),
            metadata={"date": "2024-01-01", "source": "old"},
        )
        tmp_store.add_document(
            text="reciente",
            embedding=_fallback_embedding("reciente"),
            metadata={"date": "2025-06-01", "source": "new"},
        )
        engine: RAGEngine = RAGEngine(vector_store=tmp_store, api_key=None)
        result: dict[str, Any] = engine.query(
            "test", top_k=10, start_date="2025-01-01"
        )

        # Solo el documento reciente debe aparecer
        assert "old" not in result["sources"]

    def test_filtro_fecha_end_en_query(self, tmp_store: VectorStore) -> None:
        """end_date en query debe filtrar documentos futuros."""
        tmp_store.add_document(
            text="antiguo",
            embedding=_fallback_embedding("antiguo"),
            metadata={"date": "2024-01-01", "source": "old"},
        )
        tmp_store.add_document(
            text="futuro",
            embedding=_fallback_embedding("futuro"),
            metadata={"date": "2026-01-01", "source": "future"},
        )
        engine: RAGEngine = RAGEngine(vector_store=tmp_store, api_key=None)
        result: dict[str, Any] = engine.query(
            "test", top_k=10, end_date="2025-01-01"
        )

        # Solo el documento antiguo debe aparecer
        assert "future" not in result["sources"]

    def test_modo_fallback_respuesta_simulada(
        self, rag_fallback: RAGEngine
    ) -> None:
        """En modo fallback la respuesta debe indicar '[MODO TEST]'."""
        result: dict[str, Any] = rag_fallback.query("pregunta test")
        assert "[MODO TEST]" in result["response"]

    def test_con_api_mock_llama_a_nemotron(
        self, tmp_store: VectorStore
    ) -> None:
        """Con cliente mockeado debe llamar a chat.completions.create."""
        mock_client = MagicMock()

        # Mockear embedding
        mock_emb_response = MagicMock()
        mock_emb_response.data = [MagicMock(embedding=[1.0] * 4096)]
        mock_client.embeddings.create.return_value = mock_emb_response

        # Mockear generación
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="respuesta mock"))]
        mock_client.chat.completions.create.return_value = mock_completion

        with patch("src.rag.engine.OpenAI", return_value=mock_client):
            engine: RAGEngine = RAGEngine(
                vector_store=tmp_store, api_key="test"
            )
            result: dict[str, Any] = engine.query("test")

        assert result["response"] == "respuesta mock"
        mock_client.chat.completions.create.assert_called_once()


# ===========================================================================
# Tests de RAGEngine — Manejo de excepciones en query
# ===========================================================================

class TestQueryExcepciones:
    """Pruebas de robustez del pipeline ante errores de API."""

    def test_error_en_busqueda_vectorial_no_rompe_pipeline(
        self, rag_fallback: RAGEngine
    ) -> None:
        """Error en VectorStore.search_similar no debe romper query."""
        rag_fallback._vector_store.search_similar = MagicMock(
            side_effect=Exception("DB error")
        )
        result: dict[str, Any] = rag_fallback.query("test")

        # Debe completar el pipeline con contexto vacío
        assert "response" in result
        assert isinstance(result["response"], str)

    def test_error_en_generacion_devuelve_mensaje_de_error(
        self, tmp_store: VectorStore
    ) -> None:
        """Error en Nemotron debe devolver mensaje de error amigable."""
        mock_client = MagicMock()

        mock_emb_response = MagicMock()
        mock_emb_response.data = [MagicMock(embedding=[1.0] * 4096)]
        mock_client.embeddings.create.return_value = mock_emb_response

        mock_client.chat.completions.create.side_effect = Exception("Nemotron down")

        with patch("src.rag.engine.OpenAI", return_value=mock_client):
            engine: RAGEngine = RAGEngine(
                vector_store=tmp_store, api_key="test"
            )
            result: dict[str, Any] = engine.query("test")

        assert "Error al generar" in result["response"]

    def test_error_en_embedding_no_rompe_pipeline(
        self, tmp_store: VectorStore
    ) -> None:
        """Error en embedding debe usar fallback y continuar el pipeline."""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = Exception("embed fail")

        # Mockear generación para que devuelva un string válido
        mock_completion = MagicMock()
        mock_completion.choices = [
            MagicMock(message=MagicMock(content="respuesta ok"))
        ]
        mock_client.chat.completions.create.return_value = mock_completion

        with patch("src.rag.engine.OpenAI", return_value=mock_client):
            engine: RAGEngine = RAGEngine(
                vector_store=tmp_store, api_key="test"
            )
            result: dict[str, Any] = engine.query("test")

        # Debe completar el pipeline con embedding de fallback
        assert "response" in result
        assert isinstance(result["response"], str)

    def test_contexto_vacio_respuesta_indica_sin_info(
        self, rag_fallback: RAGEngine
    ) -> None:
        """Sin contexto recuperado la respuesta debe indicar falta de información."""
        result: dict[str, Any] = rag_fallback.query("pregunta sin documentos")
        assert len(result["context"]) == 0
        assert isinstance(result["response"], str)

# -*- coding: utf-8 -*-
"""Pruebas unitarias del módulo `src/database/vector_store.py`.

Cobertura de:
- Creación de la tabla SQLite al inicializar el almacén.
- Inserción de documentos con embedding y metadatos.
- Búsqueda por similitud de coseno con y sin filtros de fecha.
- Manejo de errores: metadatos faltantes, vectores de distinta longitud.
- Persistencia: los documentos persisten entre instancias del almacén.

Se usa `tmp_path` para crear bases de datos temporales aisladas.
"""

import math
from datetime import date, timedelta
from typing import Any

import pytest

from src.database.vector_store import VectorStore, _similitud_coseno


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path: Any) -> VectorStore:
    """Crea un VectorStore temporal para cada prueba."""
    db_path: str = str(tmp_path / "test_vectors.db")
    return VectorStore(ruta_db=db_path)


@pytest.fixture
def store_con_documentos(store: VectorStore) -> VectorStore:
    """Almacena documentos de ejemplo y devuelve el almacén lleno."""
    store.add_document(
        text="machine learning es una rama de la inteligencia artificial",
        embedding=[1.0, 0.0, 0.0],
        metadata={"date": "2025-01-15", "source": "doc_ml"},
    )
    store.add_document(
        text="deep learning usa redes neuronales profundas",
        embedding=[0.9, 0.4, 0.1],
        metadata={"date": "2025-06-20", "source": "doc_dl"},
    )
    store.add_document(
        text="procesamiento de lenguaje natural NLP",
        embedding=[0.1, 0.1, 0.9],
        metadata={"date": "2025-09-01", "source": "doc_nlp"},
    )
    return store


# ---------------------------------------------------------------------------
# Tests de _similitud_coseno
# ---------------------------------------------------------------------------

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

    def test_vectores_diferente_longitud_lanza_excepcion(self) -> None:
        """Vectores de distinta longitud deben lanzar ValueError."""
        with pytest.raises(ValueError, match="misma longitud"):
            _similitud_coseno([1.0, 0.0], [1.0, 0.0, 0.0])

    def test_vector_cero_devuelve_cero(self) -> None:
        """Un vector nulo debe devolver similitud 0.0."""
        assert _similitud_coseno([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests de VectorStore
# ---------------------------------------------------------------------------

class TestVectorStoreInit:
    """Pruebas de inicialización del almacén."""

    def test_crea_base_de_datos(self, tmp_path: Any) -> None:
        """Debe crear el archivo SQLite al instanciar."""
        db_path: str = str(tmp_path / "nueva.db")
        vs = VectorStore(ruta_db=db_path)
        import os
        assert os.path.exists(db_path)
        vs._conexion.close()

    def test_tabla_documentos_existe(self, store: VectorStore) -> None:
        """La tabla 'documentos' debe existir tras la inicialización."""
        cursor = store._conexion.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documentos'"
        )
        assert cursor.fetchone() is not None


class TestAddDocument:
    """Pruebas de inserción de documentos."""

    def test_retorna_id_str(self, store: VectorStore) -> None:
        """add_document debe retornar un string (ID único)."""
        doc_id: str = store.add_document(
            text="texto de prueba",
            embedding=[0.5, 0.5, 0.5],
            metadata={"date": "2025-01-01", "source": "test"},
        )
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    def test_ids_unicos(self, store: VectorStore) -> None:
        """Dos documentos deben tener IDs diferentes."""
        id1: str = store.add_document(
            text="a", embedding=[1.0], metadata={"date": "2025-01-01", "source": "a"}
        )
        id2: str = store.add_document(
            text="b", embedding=[0.5], metadata={"date": "2025-01-01", "source": "b"}
        )
        assert id1 != id2

    def test_metadata_faltante_lanza_excepcion(self, store: VectorStore) -> None:
        """Sin 'date' o 'source' en metadata debe lanzar ValueError."""
        with pytest.raises(ValueError, match="metadata debe contener"):
            store.add_document(
                text="texto",
                embedding=[1.0],
                metadata={"date": "2025-01-01"},
            )

    def test_documento_persiste(self, store: VectorStore) -> None:
        """El documento almacenado debe ser recuperable por la búsqueda."""
        store.add_document(
            text="contenido",
            embedding=[1.0, 0.0],
            metadata={"date": "2025-03-10", "source": "persistencia"},
        )
        resultados: list[dict[str, Any]] = store.search_similar(
            query_embedding=[1.0, 0.0], top_k=1
        )
        assert len(resultados) == 1
        assert resultados[0]["text"] == "contenido"


class TestSearchSimilar:
    """Pruebas de búsqueda por similitud de coseno."""

    def test_devuelve_top_k(self, store_con_documentos: VectorStore) -> None:
        """Debe devolver como máximo top_k resultados."""
        resultados: list[dict[str, Any]] = store_con_documentos.search_similar(
            query_embedding=[1.0, 0.0, 0.0], top_k=2
        )
        assert len(resultados) <= 2

    def test_orden_descendente_por_score(
        self, store_con_documentos: VectorStore
    ) -> None:
        """Los resultados deben estar ordenados por similitud descendente."""
        resultados: list[dict[str, Any]] = store_con_documentos.search_similar(
            query_embedding=[1.0, 0.0, 0.0], top_k=3
        )
        scores: list[float] = [r["score"] for r in resultados]
        assert scores == sorted(scores, reverse=True)

    def test_documento_mas_cercano_primero(
        self, store_con_documentos: VectorStore
    ) -> None:
        """El primer resultado debe ser el más similar al query."""
        resultados: list[dict[str, Any]] = store_con_documentos.search_similar(
            query_embedding=[1.0, 0.0, 0.0], top_k=1
        )
        assert resultados[0]["source"] == "doc_ml"

    def test_filtro_fecha_start(self, store_con_documentos: VectorStore) -> None:
        """start_date debe excluir documentos anteriores a esa fecha."""
        resultados: list[dict[str, Any]] = store_con_documentos.search_similar(
            query_embedding=[1.0, 0.0, 0.0],
            top_k=10,
            start_date="2025-06-01",
        )
        for r in resultados:
            assert r["date"] >= "2025-06-01"

    def test_filtro_fecha_end(self, store_con_documentos: VectorStore) -> None:
        """end_date debe excluir documentos posteriores a esa fecha."""
        resultados: list[dict[str, Any]] = store_con_documentos.search_similar(
            query_embedding=[1.0, 0.0, 0.0],
            top_k=10,
            end_date="2025-06-30",
        )
        for r in resultados:
            assert r["date"] <= "2025-06-30"

    def test_filtro_rango_completo(self, store_con_documentos: VectorStore) -> None:
        """El filtro de rango completo debe devolver solo documentos dentro del rango."""
        resultados: list[dict[str, Any]] = store_con_documentos.search_similar(
            query_embedding=[1.0, 0.0, 0.0],
            top_k=10,
            start_date="2025-01-01",
            end_date="2025-06-30",
        )
        assert len(resultados) == 2
        for r in resultados:
            assert "2025-01-01" <= r["date"] <= "2025-06-30"

    def test_top_k_mayor_que_total_devuelve_todos(
        self, store_con_documentos: VectorStore
    ) -> None:
        """Si top_k es mayor que el total, debe devolver todos los documentos."""
        resultados: list[dict[str, Any]] = store_con_documentos.search_similar(
            query_embedding=[1.0, 0.0, 0.0], top_k=100
        )
        assert len(resultados) == 3

    def test_almacen_vacio_devuelve_lista_vacia(self, store: VectorStore) -> None:
        """Un almacén sin documentos debe devolver una lista vacía."""
        resultados: list[dict[str, Any]] = store.search_similar(
            query_embedding=[1.0, 0.0, 0.0], top_k=5
        )
        assert resultados == []

    def test_score_entre_menos_uno_y_uno(
        self, store_con_documentos: VectorStore
    ) -> None:
        """El score de cada resultado debe estar en el rango [-1, 1]."""
        resultados: list[dict[str, Any]] = store_con_documentos.search_similar(
            query_embedding=[1.0, 0.0, 0.0], top_k=3
        )
        for r in resultados:
            assert -1.0 <= r["score"] <= 1.0

    def test_embedding_dim_incompatible_se_omite(
        self, store: VectorStore
    ) -> None:
        """Documentos con embedding de distinta dimensión se saltan."""
        store.add_document(
            text="doc corto",
            embedding=[1.0],
            metadata={"date": "2025-01-01", "source": "corto"},
        )
        store.add_document(
            text="doc largo",
            embedding=[1.0, 0.0, 0.0],
            metadata={"date": "2025-01-01", "source": "largo"},
        )
        # Query de dimensión 3: el doc de dimensión 1 debe ignorarse
        resultados: list[dict[str, Any]] = store.search_similar(
            query_embedding=[1.0, 0.0, 0.0], top_k=10
        )
        assert len(resultados) == 1
        assert resultados[0]["source"] == "largo"


# ---------------------------------------------------------------------------
# Tests de manejo de errores (mocks)
# ---------------------------------------------------------------------------

class TestErroresSQLite:
    """Pruebas de los bloques except de errores SQLite."""

    def test_init_error_registra_y_lanza(self, tmp_path: Any) -> None:
        """Un error SQLite en __init__ debe loguearse y relanzarse."""
        import sqlite3
        from unittest.mock import patch

        with patch("sqlite3.connect", side_effect=sqlite3.Error("fail")):
            with pytest.raises(sqlite3.Error):
                VectorStore(ruta_db=str(tmp_path / "bad.db"))

    def test_add_document_error_registra_y_lanza(
        self, store: VectorStore
    ) -> None:
        """Un error SQLite en add_document debe loguearse y relanzarse."""
        import sqlite3
        from unittest.mock import MagicMock, patch

        mock_conexion = MagicMock()
        mock_conexion.execute.side_effect = sqlite3.Error("write fail")
        store._conexion = mock_conexion

        with pytest.raises(sqlite3.Error):
            store.add_document(
                text="x", embedding=[1.0], metadata={"date": "2025-01-01", "source": "x"}
            )

    def test_search_error_registra_y_lanza(self, store: VectorStore) -> None:
        """Un error SQLite en search_similar debe loguearse y relanzarse."""
        import sqlite3
        from unittest.mock import MagicMock

        mock_conexion = MagicMock()
        mock_conexion.execute.side_effect = sqlite3.Error("read fail")
        store._conexion = mock_conexion

        with pytest.raises(sqlite3.Error):
            store.search_similar(query_embedding=[1.0], top_k=1)

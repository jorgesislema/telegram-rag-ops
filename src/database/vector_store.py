# -*- coding: utf-8 -*-
"""Almacén de vectores para documentos RAG.

Proporciona la clase `VectorStore` que gestiona el almacenamiento y
búsqueda semántica de documentos utilizando SQLite como backend y
similitud de coseno como métrica de relevancia.

Cada documento se compone de:
- `text`: contenido textual del documento.
- `embedding`: representación vectorial (lista de floats).
- `date`: fecha de captura en formato ISO 8601.
- `source`: identificador de la fuente (nombre de archivo, URL, etc.).
"""

import json
import math
import sqlite3
import uuid
from typing import Any, Optional

from src.utils.logger import get_logger

logger: Any = get_logger(__name__)

# Sentencia SQL para crear la tabla de documentos.
_CREATE_TABLE_SQL: str = """
CREATE TABLE IF NOT EXISTS documentos (
    id          TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    embedding   TEXT NOT NULL,
    date        TEXT NOT NULL,
    source      TEXT NOT NULL
);
"""

# Sentencia SQL para insertar un documento.
_INSERT_DOC_SQL: str = """
INSERT INTO documentos (id, text, embedding, date, source)
VALUES (?, ?, ?, ?, ?);
"""

# Sentencia SQL para recuperar todos los documentos (filtro por fechas
# se aplica en Python porque SQLite no soporta comparaciones de vectores).
_SELECT_ALL_SQL: str = """
SELECT id, text, embedding, date, source FROM documentos;
"""


def _similitud_coseno(a: list[float], b: list[float]) -> float:
    """Calcula la similitud de coseno entre dos vectores.

    Args:
        a: Primer vector.
        b: Segundo vector.

    Returns:
        Valor entre -1 y 1 que indica la similitud direccional.

    Raises:
        ValueError: Si los vectores tienen longitudes diferentes.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Los vectores deben tener la misma longitud: {len(a)} vs {len(b)}"
        )

    producto_punto: float = sum(x * y for x, y in zip(a, b))
    magnitud_a: float = math.sqrt(sum(x * x for x in a))
    magnitud_b: float = math.sqrt(sum(x * x for x in b))

    if magnitud_a == 0.0 or magnitud_b == 0.0:
        return 0.0

    return producto_punto / (magnitud_a * magnitud_b)


class VectorStore:
    """Almacén de documentos con búsqueda por similitud de coseno.

    Utiliza SQLite como backend de persistencia. Los embeddings se
    serializan como JSON para almacenarlos en una columna de texto.

    Attributes:
        _ruta_db: Ruta al archivo de base de datos SQLite.
        _conexion: Conexión activa con la base de datos.
    """

    def __init__(self, ruta_db: str = "vector_store.db") -> None:
        """Inicializa el almacén y crea la tabla si no existe.

        Args:
            ruta_db: Ruta al archivo SQLite. Por defecto crea
                `vector_store.db` en el directorio de trabajo.
        """
        self._ruta_db: str = ruta_db
        try:
            self._conexion: sqlite3.Connection = sqlite3.connect(self._ruta_db)
            self._conexion.execute(_CREATE_TABLE_SQL)
            self._conexion.commit()
            logger.info("Almacén de vectores conectado a %s", self._ruta_db)
        except sqlite3.Error as exc:
            logger.error("Error al inicializar el almacén de vectores: %s", exc)
            raise

    def add_document(
        self,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> str:
        """Almacena un documento con su embedding y metadatos.

        Args:
            text: Contenido textual del documento.
            embedding: Representación vectorial del texto.
            metadata: Diccionario con metadatos. Debe contener al menos
                `date` (str ISO 8601) y `source` (str).

        Returns:
            Identificador único del documento almacenado.

        Raises:
            ValueError: Si falta `date` o `source` en metadata.
            sqlite3.Error: Si ocurre un error de persistencia.
        """
        doc_id: str = uuid.uuid4().hex

        # Validar que los metadatos obligatorios estén presentes
        fecha: str = metadata.get("date", "")
        fuente: str = metadata.get("source", "")
        if not fecha or not fuente:
            raise ValueError(
                "metadata debe contener 'date' (str ISO) y 'source' (str)"
            )

        embedding_json: str = json.dumps(embedding)

        try:
            self._conexion.execute(
                _INSERT_DOC_SQL,
                (doc_id, text, embedding_json, fecha, fuente),
            )
            self._conexion.commit()
            logger.debug("Documento %s almacenado correctamente", doc_id)
        except sqlite3.Error as exc:
            logger.error("Error al almacenar documento %s: %s", doc_id, exc)
            raise

        return doc_id

    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Busca los documentos más similares al vector de consulta.

        Calcula la similitud de coseno entre el embedding de consulta
        y todos los documentos almacenados, aplicando filtros de fecha
        opcionales antes de la comparación.

        Args:
            query_embedding: Vector de consulta.
            top_k: Número máximo de resultados a devolver.
            start_date: Fecha de inicio en formato ISO 8601 (inclusive).
                Si se omite, no se aplica filtro inferior.
            end_date: Fecha de fin en formato ISO 8601 (inclusive).
                Si se omite, no se aplica filtro superior.

        Returns:
            Lista de diccionarios ordenada por similitud descendente.
            Cada diccionario contiene: `id`, `text`, `date`, `source`,
            `score` (similitud de coseno).

        Raises:
            sqlite3.Error: Si ocurre un error de lectura.
        """
        try:
            cursor: sqlite3.Cursor = self._conexion.execute(_SELECT_ALL_SQL)
            filas: list[sqlite3.Row] = cursor.fetchall()
        except sqlite3.Error as exc:
            logger.error("Error al leer documentos: %s", exc)
            raise

        candidatos: list[dict[str, Any]] = []

        for fila in filas:
            doc_id: str = fila[0]
            text: str = fila[1]
            embedding: list[float] = json.loads(fila[2])
            fecha: str = fila[3]
            fuente: str = fila[4]

            # Aplicar filtro de fecha inferior
            if start_date and fecha < start_date:
                continue

            # Aplicar filtro de fecha superior
            if end_date and fecha > end_date:
                continue

            try:
                score: float = _similitud_coseno(query_embedding, embedding)
            except ValueError as exc:
                logger.warning(
                    "Dimensiones incompatibles en documento %s: %s", doc_id, exc
                )
                continue

            candidatos.append(
                {
                    "id": doc_id,
                    "text": text,
                    "date": fecha,
                    "source": fuente,
                    "score": score,
                }
            )

        # Ordenar por similitud descendente y devolver los top_k
        candidatos.sort(key=lambda d: d["score"], reverse=True)
        resultado: list[dict[str, Any]] = candidatos[:top_k]

        logger.debug(
            "Búsqueda completada: %d candidatos, %d devueltos",
            len(candidatos),
            len(resultado),
        )
        return resultado

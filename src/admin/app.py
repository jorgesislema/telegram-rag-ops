# -*- coding: utf-8 -*-
"""Panel de administración RAG en Streamlit.

Proporciona una interfaz web para gestionar el conocimiento del sistema RAG:
- Visualizar estadísticas y documentos almacenados.
- Inyectar nuevo conocimiento (texto + embedding + metadatos).
- Eliminar documentos existentes.

Ejecutar con:
    cd telegram-rag-ops
    streamlit run src/admin/app.py
"""

import sys
from pathlib import Path

# Asegurar que la raíz del proyecto esté en sys.path para imports absolutos
_RUTA_ADMIN: Path = Path(__file__).resolve().parent
_RUTA_RAIZ: Path = _RUTA_ADMIN.parent.parent
if str(_RUTA_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RUTA_RAIZ))

import json
import sqlite3
from datetime import date, datetime
from typing import Any

import streamlit as st

from src.database import VectorStore
from src.rag import RAGEngine
from src.utils.logger import get_logger

logger: Any = get_logger(__name__)

# Ruta a la base de datos persistente
_RUTA_BASE: Path = Path(__file__).resolve().parent.parent.parent
_DIRECTORIO_DATA: Path = _RUTA_BASE / "data"
_RUTA_DB: str = str(_DIRECTORIO_DATA / "vectors.db")

# ---------------------------------------------------------------------------
# CSS personalizado — tema oscuro con acentos neón
# ---------------------------------------------------------------------------
_CSS: str = """
<style>
    /* Fondo general oscuro */
    .stApp {
        background-color: #0e1117;
    }

    /* Encabezados principales */
    h1, h2, h3 {
        color: #00d4ff !important;
    }

    /* Subtítulos y labels */
    .stMarkdown p, .stMarkdown li, label {
        color: #c9d1d9 !important;
    }

    /* Métricas / stats */
    [data-testid="stMetricValue"] {
        color: #39ff14 !important;
        font-size: 1.8rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
    }

    /* Botones primarios */
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button {
        background-color: #1a7f37 !important;
        color: #ffffff !important;
        border: 1px solid #39ff14 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button:hover {
        background-color: #238636 !important;
        border-color: #00d4ff !important;
    }

    /* Botones de eliminación */
    .stButton > button[kind="secondary"] {
        background-color: #da3633 !important;
        color: #ffffff !important;
        border: 1px solid #f85149 !important;
        border-radius: 6px !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #b62324 !important;
    }

    /* Tablas */
    .stDataFrame {
        border: 1px solid #30363d !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
    }
    [data-testid="stSidebar"] h2 {
        color: #39ff14 !important;
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stDateInput > div > div > input {
        background-color: #0d1117 !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
    }

    /* Separadores */
    hr {
        border-color: #21262d !important;
    }
</style>
"""


# ---------------------------------------------------------------------------
# Funciones de acceso a datos
# ---------------------------------------------------------------------------

def _obtener_conexion(ruta_db: str) -> sqlite3.Connection:
    """Devuelve una conexión SQLite a la base de datos indicada."""
    return sqlite3.connect(ruta_db)


def _contar_documentos(conexion: sqlite3.Connection) -> int:
    """Cuenta el total de documentos almacenados."""
    cursor: sqlite3.Cursor = conexion.execute("SELECT COUNT(*) FROM documentos")
    return int(cursor.fetchone()[0])


def _listar_documentos(conexion: sqlite3.Connection) -> list[dict[str, Any]]:
    """Recupera todos los documentos (sin embedding) para mostrar en tabla."""
    cursor: sqlite3.Cursor = conexion.execute(
        "SELECT id, text, date, source FROM documentos ORDER BY date DESC"
    )
    filas: list[sqlite3.Row] = cursor.fetchall()
    return [
        {
            "id": fila[0],
            "texto": fila[1][:120] + ("..." if len(fila[1]) > 120 else ""),
            "fecha": fila[2],
            "fuente": fila[3],
        }
        for fila in filas
    ]


def _obtener_fuentes_unicas(conexion: sqlite3.Connection) -> list[str]:
    """Devuelve las fuentes únicas registradas."""
    cursor: sqlite3.Cursor = conexion.execute(
        "SELECT DISTINCT source FROM documentos ORDER BY source"
    )
    return [fila[0] for fila in cursor.fetchall()]


def _eliminar_documento(conexion: sqlite3.Connection, doc_id: str) -> int:
    """Elimina un documento por su ID. Retorna filas afectadas."""
    cursor: sqlite3.Cursor = conexion.execute(
        "DELETE FROM documentos WHERE id = ?", (doc_id,)
    )
    conexion.commit()
    return cursor.rowcount


# ---------------------------------------------------------------------------
# Inicialización de sesión
# ---------------------------------------------------------------------------

def _inicializar_sesion() -> None:
    """Crea objetos compartidos en el estado de la sesión de Streamlit."""
    if "vector_store" not in st.session_state:
        _DIRECTORIO_DATA.mkdir(parents=True, exist_ok=True)
        st.session_state.vector_store = VectorStore(ruta_db=_RUTA_DB)
        logger.info("VectorStore inicializado en %s", _RUTA_DB)

    if "rag_engine" not in st.session_state:
        api_key: str = st.session_state.get("nvidia_api_key", "")
        st.session_state.rag_engine = RAGEngine(
            vector_store=st.session_state.vector_store,
            api_key=api_key if api_key else None,
        )
        logger.info("RAGEngine inicializado")


# ---------------------------------------------------------------------------
# Panel principal
# ---------------------------------------------------------------------------

def _panel_principal(conexion: sqlite3.Connection) -> None:
    """Renderiza el panel principal con estadísticas y tabla de documentos."""
    st.title("Panel de Administracion RAG")
    st.markdown("---")

    # --- Estadisticas ---
    total: int = _contar_documentos(conexion)
    fuentes: list[str] = _obtener_fuentes_unicas(conexion)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Documentos", value=total)
    with col2:
        st.metric(label="Fuentes", value=len(fuentes))
    with col3:
        estado_db: str = "Conectada" if conexion else "Desconectada"
        st.metric(label="Base de Datos", value=estado_db)

    st.markdown("---")

    # --- Tabla de documentos ---
    st.subheader("Documentos Almacenados")
    docs: list[dict[str, Any]] = _listar_documentos(conexion)

    if not docs:
        st.info("No hay documentos en la base de conocimiento.")
        return

    st.dataframe(
        docs,
        use_container_width=True,
        height=400,
        column_config={
            "id": st.column_config.TextColumn("ID", width="small"),
            "texto": st.column_config.TextColumn("Texto", width="large"),
            "fecha": st.column_config.TextColumn("Fecha", width="medium"),
            "fuente": st.column_config.TextColumn("Fuente", width="medium"),
        },
    )

    # --- Eliminar documento ---
    st.markdown("---")
    st.subheader("Eliminar Documento")
    ids_disponibles: list[str] = [d["id"] for d in docs]
    id_seleccionado: str = st.selectbox(
        "Selecciona el ID del documento a eliminar",
        options=ids_disponibles,
        key="select_eliminar",
    )
    if st.button("Eliminar Documento", type="secondary", key="btn_eliminar"):
        filas: int = _eliminar_documento(conexion, id_seleccionado)
        if filas > 0:
            st.success(f"Documento {id_seleccionado[:12]}... eliminado.")
            logger.info(
                "Documento eliminado por administrador: %s", id_seleccionado
            )
            st.rerun()
        else:
            st.error("No se pudo eliminar el documento.")


# ---------------------------------------------------------------------------
# Sidebar — Inyección de conocimiento
# ---------------------------------------------------------------------------

def _sidebar_inyeccion() -> None:
    """Formulario en la barra lateral para inyectar nuevo conocimiento."""
    with st.sidebar:
        st.header("Inyectar Conocimiento")
        st.markdown("Agrega nuevo texto al sistema RAG.")

        with st.form("form_inyectar", clear_on_submit=True):
            texto: str = st.text_area(
                "Texto del documento",
                height=200,
                placeholder="Escribe o pega el contenido aquí...",
            )
            fecha_doc: date = st.date_input(
                "Fecha del documento",
                value=date.today(),
            )
            fuente: str = st.text_input(
                "Fuente",
                placeholder="Ej: archivo.pdf, wiki, manual...",
            )

            enviado: bool = st.form_submit_button(
                "Guardar Documento", type="primary"
            )

            if enviado:
                _procesar_inyeccion(texto, fecha_doc, fuente)


def _procesar_inyeccion(texto: str, fecha_doc: date, fuente: str) -> None:
    """Valida, genera embedding y almacena el documento."""
    # Validaciones
    if not texto.strip():
        st.error("El texto no puede estar vacio.")
        return
    if not fuente.strip():
        st.error("Debes indicar una fuente.")
        return

    # Generar embedding
    engine: RAGEngine = st.session_state.rag_engine
    try:
        embedding: list[float] = engine.generate_embedding(texto)
    except Exception as exc:
        st.error(f"Error al generar embedding: {exc}")
        logger.error("Error generando embedding: %s", exc)
        return

    # Almacenar en VectorStore
    store: VectorStore = st.session_state.vector_store
    fecha_iso: str = fecha_doc.isoformat()
    try:
        doc_id: str = store.add_document(
            text=texto,
            embedding=embedding,
            metadata={"date": fecha_iso, "source": fuente.strip()},
        )
        st.success(f"Documento guardado. ID: {doc_id[:12]}...")
        logger.info(
            "Documento inyectado por administrador: %s (fuente: %s)",
            doc_id,
            fuente,
        )
    except Exception as exc:
        st.error(f"Error al guardar: {exc}")
        logger.error("Error guardando documento: %s", exc)
        return


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Punto de entrada de la aplicacion Streamlit."""
    # Configuracion de pagina
    st.set_page_config(
        page_title="Admin RAG",
        page_icon=":gear:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Aplicar CSS personalizado
    st.markdown(_CSS, unsafe_allow_html=True)

    # Inicializar componentes
    _inicializar_sesion()
    conexion: sqlite3.Connection = _obtener_conexion(_RUTA_DB)

    # Sidebar
    _sidebar_inyeccion()

    # Panel principal
    _panel_principal(conexion)


if __name__ == "__main__":
    main()

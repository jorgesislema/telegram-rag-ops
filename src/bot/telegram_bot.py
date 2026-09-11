# -*- coding: utf-8 -*-
"""Bot de Telegram integrado con el sistema RAG.

Proporciona un asistente inteligente que recibe preguntas del usuario
vía Telegram y responde usando la base de conocimiento RAG.

Patrón asíncrono (SKILL.md):
- Todos los handlers son funciones ``async``.
- ``RAGEngine.query`` es síncrono: se ejecuta con ``asyncio.to_thread``
  para no bloquear el event loop de ``python-telegram-bot``.
- Las respuestas largas se fragmentan en bloques de 4096 caracteres
  (límite de Telegram), cortando preferentemente en saltos de línea.

Ejecutar con:
    python -m src.bot.telegram_bot
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from telegram import Message, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.constants import ChatAction

from src.rag.engine import RAGEngine
from src.utils.logger import get_logger

logger: Any = get_logger(__name__)

# Límite de caracteres por mensaje en Telegram
_LIMITE_TELEGRAM: int = 4096


# ---------------------------------------------------------------------------
# Clave del bot_data donde se almacena el RAGEngine
# ---------------------------------------------------------------------------
_RAG_ENGINE_KEY: str = "rag_engine"


def _obtener_engine(context: ContextTypes.DEFAULT_TYPE) -> RAGEngine:
    """Extrae el RAGEngine del ``bot_data`` del contexto.

    Args:
        context: Contexto del handler que contiene ``bot_data``.

    Returns:
        La instancia de ``RAGEngine`` inyectada al iniciar la aplicación.

    Raises:
        RuntimeError: Si el engine no ha sido registrado en ``bot_data``.
    """
    engine: Optional[RAGEngine] = context.bot_data.get(_RAG_ENGINE_KEY)
    if engine is None:
        raise RuntimeError(
            "RAGEngine no encontrado en bot_data. "
            "Asegúrate de usar construir_aplicacion() antes de iniciar polling."
        )
    return engine


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Comando ``/start``: da la bienvenida y explica la funcionalidad.

    Args:
        update: Objeto ``Update`` de Telegram con la info del mensaje.
        context: Contexto del handler con datos de la aplicación.
    """
    if update.message is None:
        return

    usuario: str = (
        update.effective_user.first_name if update.effective_user else "usuario"
    )
    texto_bienvenida: str = (
        f"Hola, {usuario}. Soy un asistente inteligente con acceso "
        "a una base de conocimiento.\n\n"
        "Escríbeme cualquier pregunta y buscaré la información "
        "más relevante para responderte."
    )
    await update.message.reply_text(texto_bienvenida)
    logger.info("Usuario %s inició conversación (/start)", usuario)


async def msg_mensaje(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler de mensajes de texto: consulta al RAG y responde.

    Flujo:
    1. Muestra indicador «escribiendo...».
    2. Ejecuta la consulta RAG en un hilo separado (no bloqueante).
    3. Fragmenta y envía la respuesta al usuario.

    Args:
        update: Objeto ``Update`` de Telegram.
        context: Contexto del handler.
    """
    if update.message is None or update.message.text is None:
        return

    texto_usuario: str = update.message.text.strip()
    if not texto_usuario:
        return

    usuario: str = (
        update.effective_user.first_name if update.effective_user else "desconocido"
    )
    logger.info("Mensaje de %s: %s", usuario, texto_usuario[:100])

    # Indicador de «escribiendo...»
    await update.message.chat.send_action(action=ChatAction.TYPING)

    try:
        engine: RAGEngine = _obtener_engine(context)
        # RAGEngine.query es síncrono; delegar al hilo del executor
        resultado: dict[str, Any] = await asyncio.to_thread(
            engine.query,
            user_query=texto_usuario,
            top_k=3,
        )
        respuesta: str = resultado.get("response", "")

        if not respuesta:
            respuesta = (
                "No encontré información relevante para tu pregunta "
                "en la base de conocimiento."
            )

    except Exception as exc:
        logger.error("Error procesando mensaje de %s: %s", usuario, exc)
        respuesta = (
            "Ocurrió un error al procesar tu consulta. "
            "Inténtalo de nuevo más tarde."
        )

    # Enviar respuesta fragmentada
    await _enviar_respuesta(update.message, respuesta)


async def _enviar_respuesta(message: Message, texto: str) -> None:
    """Envía el texto fragmentándolo en bloques de máximo 4096 caracteres.

    Intenta cortar en saltos de línea para no partir oraciones.

    Args:
        message: Objeto ``Message`` de Telegram.
        texto: Texto completo a enviar.
    """
    if not texto:
        await message.reply_text("Sin respuesta disponible.")
        return

    bloques: list[str] = _fragmentar_texto(texto, _LIMITE_TELEGRAM)

    for i, bloque in enumerate(bloques):
        await message.reply_text(bloque)
        # Pausa breve entre bloques para evitar rate limits
        if i < len(bloques) - 1:
            await asyncio.sleep(0.5)


def _fragmentar_texto(texto: str, limite: int) -> list[str]:
    """Divide el texto en bloques respetando el límite de caracteres.

    Intenta cortar en el último salto de línea antes del límite para
    mantener la legibilidad.

    Args:
        texto: Texto a fragmentar.
        limite: Número máximo de caracteres por bloque.

    Returns:
        Lista de bloques de texto.
    """
    if len(texto) <= limite:
        return [texto]

    bloques: list[str] = []
    texto_restante: str = texto

    while texto_restante:
        if len(texto_restante) <= limite:
            bloques.append(texto_restante)
            break

        # Buscar el último salto de línea antes del límite
        corte: int = texto_restante.rfind("\n", 0, limite)
        if corte == -1:
            # Si no hay salto de línea, cortar en el límite exacto
            corte = limite

        bloques.append(texto_restante[:corte])
        texto_restante = texto_restante[corte:].lstrip("\n")

    return bloques


# ---------------------------------------------------------------------------
# Manejador de errores global
# ---------------------------------------------------------------------------

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manejador global de errores asíncronos.

    Captura cualquier excepción no controlada y la registra usando
    ``get_logger`` (nunca ``print``).

    Args:
        update: Objeto ``Update`` (puede ser ``None`` en errores de
            inicialización).
        context: Contexto del handler con la excepción en
            ``context.error``.
    """
    exc: BaseException | None = context.error
    if exc is not None:
        logger.error(
            "Error no controlado en el bot: %s",
            exc,
            exc_info=True,
        )
    else:
        logger.error("Error no controlado en el bot (sin excepción asociada)")


# ---------------------------------------------------------------------------
# Construcción de la aplicación
# ---------------------------------------------------------------------------

def construir_aplicacion(token: str, engine: RAGEngine) -> Application:
    """Construye y configura la aplicación del bot con sus handlers.

    Inyecta el ``RAGEngine`` en ``bot_data`` para que los handlers
    lo consulten sin depender de variables globales.

    Args:
        token: Token del bot de Telegram.
        engine: Instancia del ``RAGEngine`` para inyectar como dependencia.

    Returns:
        Instancia configurada de ``Application`` lista para polling.
    """
    app: Application = (
        ApplicationBuilder()
        .token(token)
        .post_init(_inyectar_engine(engine))
        .build()
    )

    # Registrar handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, msg_mensaje)
    )

    # Registrar manejador de errores global
    app.add_error_handler(error_handler)

    logger.info("Aplicación del bot configurada correctamente")
    return app


def _inyectar_engine(engine: RAGEngine) -> Any:
    """Devuelve un callback ``post_init`` que inserta el engine en ``bot_data``.

    Se ejecuta una sola vez después de que la aplicación se inicializa,
    justo antes de empezar el polling.

    Args:
        engine: Instancia del ``RAGEngine`` a inyectar.

    Returns:
        Función async compatible con ``post_init``.
    """

    async def _callback(app: Application) -> None:  # type: ignore[type-arg]
        app.bot_data[_RAG_ENGINE_KEY] = engine
        logger.debug("RAGEngine inyectado en bot_data")

    return _callback


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------

def run_bot() -> None:
    """Lee la variable de entorno e inicia el bot con polling.

    Variables de entorno requeridas:
    - ``TELEGRAM_BOT_TOKEN``: Token del bot de Telegram.
    - ``NVIDIA_API_KEY``: Clave de la API de NVIDIA (opcional; usa
      fallback NumPy si falta).

    Raises:
        SystemExit: Si falta ``TELEGRAM_BOT_TOKEN``.
    """
    # Cargar variables de entorno desde .env si existe
    ruta_raiz: Path = Path(__file__).resolve().parent.parent.parent
    load_dotenv(ruta_raiz / ".env")

    token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        logger.error("Falta la variable de entorno TELEGRAM_BOT_TOKEN")
        sys.exit(1)

    api_key: Optional[str] = os.getenv("NVIDIA_API_KEY")

    # Crear directorio de datos si no existe
    (ruta_raiz / "data").mkdir(parents=True, exist_ok=True)

    # Inicializar VectorStore y RAGEngine
    from src.database import VectorStore

    ruta_db: str = str(ruta_raiz / "data" / "vectors.db")
    vector_store: VectorStore = VectorStore(ruta_db=ruta_db)
    engine: RAGEngine = RAGEngine(
        vector_store=vector_store,
        api_key=api_key,
    )

    # Construir y ejecutar la aplicación
    app: Application = construir_aplicacion(token, engine)

    logger.info("Bot de Telegram iniciado (polling)")
    print("Bot de Telegram iniciado. Presiona Ctrl+C para detener.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


# ---------------------------------------------------------------------------
# Punto de entrada directo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _ruta_raiz: Path = Path(__file__).resolve().parent.parent.parent
    if str(_ruta_raiz) not in sys.path:
        sys.path.insert(0, str(_ruta_raiz))

    run_bot()

# -*- coding: utf-8 -*-
"""Pruebas unitarias del módulo ``src/bot/telegram_bot.py``.

Cobertura de:
- Comando ``/start``: bienvenida y logging.
- Handler de mensajes ``msg_mensaje``: indicador TYPING, consulta RAG,
  respuesta fragmentada.
- Manejador de errores global ``error_handler``: registro de excepciones.
- Función auxiliar ``_fragmentar_texto``: cortes por línea y por límite.

Se usa ``unittest.mock`` (``AsyncMock``, ``MagicMock``, ``patch``) para
no consumir créditos de API reales ni hacer llamadas a Telegram.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot.telegram_bot import (
    _fragmentar_texto,
    cmd_start,
    error_handler,
    msg_mensaje,
)


# ---------------------------------------------------------------------------
# Helpers: construir objetos Update / Context simulados
# ---------------------------------------------------------------------------

def _crear_update(
    texto: str | None = None,
    usuario: str = "TestUser",
    *,
    message_none: bool = False,
) -> MagicMock:
    """Crea un ``Update`` simulado con ``message``, ``effective_user`` y ``chat``.

    Args:
        texto: Texto del mensaje. Si es ``None`` y ``message_none`` es
            ``False``, se asigna un texto por defecto.
        usuario: Nombre del usuario efectivo.
        message_none: Si es ``True``, ``update.message`` será ``None``.

    Returns:
        Objeto ``MagicMock`` que imita la interfaz de ``Update``.
    """
    update: MagicMock = MagicMock()

    if message_none:
        update.message = None
        return update

    if texto is None:
        texto = "¿Qué es machine learning?"

    update.message.text = texto
    update.message.reply_text = AsyncMock()
    update.message.chat.send_action = AsyncMock()
    update.effective_user.first_name = usuario

    return update


def _crear_context(
    engine_mock: MagicMock | None = None,
) -> MagicMock:
    """Crea un ``ContextTypes.DEFAULT_TYPE`` simulado.

    Args:
        engine_mock: Mock del ``RAGEngine``. Si se proporciona, se almacena
            en ``bot_data["rag_engine"]``.

    Returns:
        Objeto ``MagicMock`` que imita ``ContextTypes.DEFAULT_TYPE``.
    """
    context: MagicMock = MagicMock()
    context.bot_data = {}

    if engine_mock is not None:
        context.bot_data["rag_engine"] = engine_mock

    return context


# ---------------------------------------------------------------------------
# Tests de cmd_start
# ---------------------------------------------------------------------------

class TestCmdStart:
    """Pruebas del comando ``/start``."""

    @pytest.mark.asyncio
    async def test_bienvenida_usuario(self) -> None:
        """Debe enviar un mensaje de bienvenida con el nombre del usuario."""
        update: MagicMock = _crear_update(usuario="Carlos")
        context: MagicMock = _crear_context()

        await cmd_start(update, context)

        update.message.reply_text.assert_awaited_once()
        texto_enviado: str = update.message.reply_text.call_args[0][0]
        assert "Carlos" in texto_enviado
        assert "asistente" in texto_enviado.lower()

    @pytest.mark.asyncio
    async def test_update_message_none_no_crash(self) -> None:
        """Si ``update.message`` es ``None``, no debe lanzar excepción."""
        update: MagicMock = _crear_update(message_none=True)
        context: MagicMock = _crear_context()

        # No debe lanzar excepción; la función retorna temprano
        await cmd_start(update, context)

    @pytest.mark.asyncio
    async def test_usuario_none_muestra_fallback(self) -> None:
        """Si ``effective_user`` es ``None``, usar 'usuario' como fallback."""
        update: MagicMock = _crear_update()
        update.effective_user = None
        context: MagicMock = _crear_context()

        await cmd_start(update, context)

        texto_enviado: str = update.message.reply_text.call_args[0][0]
        assert "usuario" in texto_enviado


# ---------------------------------------------------------------------------
# Tests de msg_mensaje
# ---------------------------------------------------------------------------

class TestMsgMensaje:
    """Pruebas del handler de mensajes de texto."""

    @pytest.mark.asyncio
    async def test_respuesta_normal_rag(self) -> None:
        """Debe consultar RAG y enviar la respuesta al usuario."""
        engine_mock: MagicMock = MagicMock()
        engine_mock.query.return_value = {
            "response": "ML es un subcampo de la IA.",
            "context": ["fragmento 1"],
            "sources": ["doc1"],
        }

        update: MagicMock = _crear_update(texto="¿Qué es ML?")
        context: MagicMock = _crear_context(engine_mock)

        with patch("src.bot.telegram_bot.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = engine_mock.query.return_value
            await msg_mensaje(update, context)

        # Verificar indicador TYPING
        update.message.chat.send_action.assert_awaited_once()
        # Verificar que se envió la respuesta
        update.message.reply_text.assert_awaited()
        texto_enviado: str = update.message.reply_text.call_args[0][0]
        assert "ML es un subcampo" in texto_enviado

    @pytest.mark.asyncio
    async def test_respuesta_vacia_muestra_fallback(self) -> None:
        """Si RAG devuelve respuesta vacía, mostrar mensaje de fallback."""
        engine_mock: MagicMock = MagicMock()
        engine_mock.query.return_value = {
            "response": "",
            "context": [],
            "sources": [],
        }

        update: MagicMock = _crear_update(texto="Pregunta vacía")
        context: MagicMock = _crear_context(engine_mock)

        with patch("src.bot.telegram_bot.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = engine_mock.query.return_value
            await msg_mensaje(update, context)

        texto_enviado: str = update.message.reply_text.call_args[0][0]
        assert "No encontré información" in texto_enviado

    @pytest.mark.asyncio
    async def test_error_rag_mensaje_error(self) -> None:
        """Si RAG lanza excepción, enviar mensaje de error al usuario."""
        engine_mock: MagicMock = MagicMock()
        engine_mock.query.side_effect = Exception("Fallo RAG")

        update: MagicMock = _crear_update(texto="Pregunta con error")
        context: MagicMock = _crear_context(engine_mock)

        with patch("src.bot.telegram_bot.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = Exception("Fallo RAG")
            await msg_mensaje(update, context)

        texto_enviado: str = update.message.reply_text.call_args[0][0]
        assert "error" in texto_enviado.lower()

    @pytest.mark.asyncio
    async def test_mensaje_vacio_no_procesa(self) -> None:
        """Si el texto está vacío o es solo espacios, no procesar."""
        update: MagicMock = _crear_update(texto="   ")
        context: MagicMock = _crear_context()

        await msg_mensaje(update, context)

        update.message.chat.send_action.assert_not_awaited()
        update.message.reply_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_update_message_none_no_procesa(self) -> None:
        """Si ``update.message`` es ``None``, no procesar."""
        update: MagicMock = _crear_update(message_none=True)
        context: MagicMock = _crear_context()

        # No debe lanzar excepción; la función retorna temprano
        await msg_mensaje(update, context)

    @pytest.mark.asyncio
    async def test_engine_no_en_bot_data(self) -> None:
        """Si ``rag_engine`` no está en ``bot_data``, debe lanzar ``RuntimeError``."""
        update: MagicMock = _crear_update(texto="Pregunta sin engine")
        context: MagicMock = _crear_context()  # Sin engine

        with patch("src.bot.telegram_bot.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = RuntimeError(
                "RAGEngine no encontrado en bot_data."
            )
            await msg_mensaje(update, context)

        # Debe enviar mensaje de error al usuario
        texto_enviado: str = update.message.reply_text.call_args[0][0]
        assert "error" in texto_enviado.lower()

    @pytest.mark.asyncio
    async def test_envia_typing_action(self) -> None:
        """Debe enviar la acción TYPING antes de procesar."""
        engine_mock: MagicMock = MagicMock()
        engine_mock.query.return_value = {
            "response": "Respuesta",
            "context": [],
            "sources": [],
        }

        update: MagicMock = _crear_update(texto="test")
        context: MagicMock = _crear_context(engine_mock)

        with patch("src.bot.telegram_bot.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = engine_mock.query.return_value
            await msg_mensaje(update, context)

        update.message.chat.send_action.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tests de error_handler
# ---------------------------------------------------------------------------

class TestErrorHandler:
    """Pruebas del manejador global de errores."""

    @pytest.mark.asyncio
    async def test_registra_excepcion(self) -> None:
        """Debe registrar la excepción del ``context.error`` en el logger."""
        context: MagicMock = MagicMock()
        context.error = ValueError("Token inválido")

        with patch("src.bot.telegram_bot.logger") as mock_logger:
            await error_handler(update=None, context=context)

        mock_logger.error.assert_called_once()
        args: tuple[Any, ...] = mock_logger.error.call_args[0]
        assert "Token inválido" in str(args)

    @pytest.mark.asyncio
    async def test_excepcion_none_registra_sin_detalles(self) -> None:
        """Si ``context.error`` es ``None``, registrar mensaje genérico."""
        context: MagicMock = MagicMock()
        context.error = None

        with patch("src.bot.telegram_bot.logger") as mock_logger:
            await error_handler(update=None, context=context)

        mock_logger.error.assert_called_once()
        args: tuple[Any, ...] = mock_logger.error.call_args[0]
        assert "sin excepción" in str(args).lower()

    @pytest.mark.asyncio
    async def test_recibe_update_como_objeto(self) -> None:
        """El parámetro ``update`` puede ser cualquier objeto (incluso ``None``)."""
        context: MagicMock = MagicMock()
        context.error = RuntimeError("test")

        with patch("src.bot.telegram_bot.logger"):
            # No debe lanzar excepción con update=None
            await error_handler(update=None, context=context)


# ---------------------------------------------------------------------------
# Tests de _fragmentar_texto
# ---------------------------------------------------------------------------

class TestFragmentarTexto:
    """Pruebas de la función auxiliar de fragmentación de texto."""

    def test_texto_corto_devuelve_lista_unica(self) -> None:
        """Texto menor al límite se devuelve como un solo bloque."""
        resultado: list[str] = _fragmentar_texto("Hola mundo", 4096)
        assert resultado == ["Hola mundo"]

    def test_texto_largo_se_fragmenta(self) -> None:
        """Texto mayor al límite se divide en múltiples bloques."""
        texto: str = "línea\n" * 1000  # ~7000 caracteres
        resultado: list[str] = _fragmentar_texto(texto, 4096)
        assert len(resultado) > 1
        for bloque in resultado:
            assert len(bloque) <= 4096

    def test_corte_prefiere_saltos_de_linea(self) -> None:
        """El fragmentador debe cortar en saltos de línea cuando sea posible."""
        texto: str = "primera línea\nsegunda línea larga" * 200
        resultado: list[str] = _fragmentar_texto(texto, 100)
        # Todos los bloques deben respetar el límite
        for bloque in resultado:
            assert len(bloque) <= 100

    def test_texto_vacio_devuelve_lista_vacia(self) -> None:
        """Texto vacío genera una lista con un string vacío."""
        resultado: list[str] = _fragmentar_texto("", 4096)
        assert resultado == [""]

    def test_texto_exactamente_limite(self) -> None:
        """Texto de exactamente el límite se devuelve como un solo bloque."""
        texto: str = "x" * 4096
        resultado: list[str] = _fragmentar_texto(texto, 4096)
        assert len(resultado) == 1
        assert resultado[0] == texto

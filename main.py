# -*- coding: utf-8 -*-
"""Punto de entrada principal de telegram-rag-ops.

Carga las variables de entorno, configura el logger y arranca el bot
de Telegram con su motor RAG.

Ejecutar con:
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# Asegurar que la raíz del proyecto está en sys.path para que
# las importaciones de src.* funcionen tanto con `python main.py`
# como con `python -m src.bot.telegram_bot`.
_RUTA_RAIZ: Path = Path(__file__).resolve().parent
if str(_RUTA_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RUTA_RAIZ))

# Cargar variables de entorno desde .env antes de cualquier importación
# que dependa de ellas (logger, RAGEngine, etc.)
load_dotenv(_RUTA_RAIZ / ".env")

from src.utils.logger import get_logger  # noqa: E402

logger: object = get_logger(__name__)


if __name__ == "__main__":
    try:
        from src.bot.telegram_bot import run_bot

        logger.info("Iniciando telegram-rag-ops")
        run_bot()
    except KeyboardInterrupt:
        logger.info("Bot detenido por el usuario (Ctrl+C)")
    except SystemExit:
        # sys.exit() dentro de run_bot ya habrá registrado el error
        raise
    except Exception as exc:
        logger.critical("Excepción no controlada al iniciar el bot: %s", exc, exc_info=True)
        sys.exit(1)

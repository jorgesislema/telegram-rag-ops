# -*- coding: utf-8 -*-
"""Bot de Telegram (solo interfaz).

El módulo ``telegram_bot`` se importa bajo demanda para no obligar
a instalar ``python-telegram-bot`` cuando solo se usa la capa RAG.
"""

from src.bot.telegram_bot import construir_aplicacion, run_bot

__all__: list[str] = ["construir_aplicacion", "run_bot"]

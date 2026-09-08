# AGENTS.md — telegram-rag-ops

Reglas obligatorias para todos los agentes (IA o humanos) que trabajen en este repositorio.

## Objetivo del Proyecto

Sistema de RAG (Retrieval-Augmented Generation) integrado con:
- **Telegram** (bot de interacción con usuarios, `python-telegram-bot`)
- **Streamlit** (panel de administración, `src/admin`)
- **Nemotron** (modelo LLM de NVIDIA vía API compatible con OpenAI)

## Reglas de Arquitectura

- **Tipado estricto:** Todo el código debe usar type hints del módulo `typing` de Python (funciones, métodos, retornos y atributos de clase).
- **Diseño modular:** Separación clara de responsabilidades:
  - `src/bot` — Lógica del bot de Telegram (solo interfaz)
  - `src/rag` — Lógica RAG: embeddings, recuperación, generación (solo negocio)
  - `src/admin` — Panel de administración en Streamlit (solo interfaz)
  - `src/database` — Acceso y persistencia de datos (solo negocio)
  - `src/utils` — Utilidades transversales (logger, config, helpers)
- **Prohibición absoluta de importación cruzada** entre la capa de interfaz (`src/bot`, `src/admin`) y la lógica de negocio (`src/rag`, `src/database`). La comunicación solo ocurre a través de contratos definidos en la capa de negocio.

## Gestión de Errores y Calidad

- Toda llamada externa (API de Telegram, API de NVIDIA/Nemotron, embeddings) debe envolverse en bloques `try/except` que registren el error consumiendo el módulo `src/utils/logger.py`.
- **Prohibido:** `except Exception: pass` vacíos y reportar errores con `print()`. Siempre usa el logger.

## Estándar de Pruebas (>90% de cobertura)

- Toda función de lógica de negocio (`src/rag`, `src/database`) debe tener sus pruebas correspondientes en `tests/`.
- **Obligatorio usar mocks** (por ejemplo, `unittest.mock` / `pytest-mock`) para las llamadas a la API de Telegram y a los modelos de embeddings/LLM. Nunca consumas créditos de API reales durante las pruebas.
- Ejecutar `pytest --cov=src --cov-report=term-missing` y garantizar **cobertura ≥ 90%** antes de finalizar cualquier tarea.

## Estilo de Código

- **No usar iconos/emojis en el código** a menos que sea estrictamente necesario (por ejemplo, mensajes de UI que el usuario final deba ver).
- **Comentarios detallados en español:** Documenta la intención de funciones, clases y bloques de lógica compleja.
- Todo texto de UI, logs y docstrings en español.

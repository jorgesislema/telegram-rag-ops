---
name: telegram-bot-rag
description: Guía y patrones para la integración de bots de Telegram asíncronos con RAG en Python.
---

# SKILL.md — telegram-bot-rag

Guía y patrones para construir bots de Telegram asíncronos integrados con un sistema RAG (embeddings + Nemotron) en Python.

## Manejo Asíncrono (asyncio)

`python-telegram-bot` v20+ está basado totalmente en `asyncio`. Todos los handlers deben ser funciones `async`.

### Patrón correcto con ApplicationBuilder y CommandHandler

```python
# Patrón correcto: registrar comandos asíncronos en la aplicación
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

async def start(update, context) -> None:
    """Comando /start: da la bienvenida al usuario del bot de Telegram."""
    await update.message.reply_text("¡Hola! Pregúntame lo que necesites.")

async def handle_message(update, context) -> None:
    """Handler principal: recibe el mensaje del usuario y consulta al RAG."""
    respuesta: str = await consultar_rag(update.message.text)
    await enviar_respuesta(update.message, respuesta)

def construir_aplicacion(token: str):
    """Construye y configura la aplicación del bot con sus handlers."""
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    return app
```

### Gestión de peticiones concurrentes sin bloquear

- **Nunca uses llamadas bloqueantes** (`requests`, `time.sleep`, I/O síncrono de BD) dentro de un handler: congelan el event loop y afectan a todos los usuarios.
- Usa clientes asíncronos: `httpx.AsyncClient` / `openai.AsyncOpenAI` para las llamadas a Nemotron y embeddings.
- Si necesitas CPU-bound o librerías síncronas inevitables, delega con `asyncio.to_thread(...)`.
- `python-telegram-bot` ya despacha cada update concurrentemente: no es necesario crear tareas manuales por usuario; solo evita bloquear.

## Flujo de Comunicación con el RAG

Secuencia completa desde que llega un mensaje de Telegram hasta que se envía la respuesta:

1. **Recepción:** El handler de Telegram (`src/bot`) recibe el update y extrae el texto del mensaje.
2. **Embedding de la consulta:** Envía la consulta al módulo de embeddings (NVIDIA) para convertirla en vector.
3. **Recuperación:** Consulta la base de datos vectorial (`src/database`) para obtener los fragmentos (chunks) relevantes por similitud.
4. **Construcción del prompt:** Combina los fragmentos recuperados + la pregunta del usuario en un prompt para Nemotron.
5. **Generación:** Llama a Nemotron para producir la respuesta final.
6. **Envío:** Fragmenta la respuesta si excede el límite de Telegram y la envía por bloques.

```python
# Ejemplo del flujo completo (capa de negocio, sin importaciones de Telegram)
async def consultar_rag(pregunta: str) -> str:
    """Ejecuta el pipeline RAG completo: embedding -> recuperación -> generación."""
    vector: list[float] = await generar_embedding(pregunta)
    fragmentos: list[str] = await recuperar_fragmentos(vector, top_k=4)
    prompt: str = construir_prompt(pregunta, fragmentos)
    respuesta: str = await generar_respuesta(prompt)
    return respuesta
```

**Importante:** el flujo RAG vive en `src/rag`. El bot de Telegram solo orquesta y envía; nunca implementa lógica de negocio (ver `AGENTS.md`).

## Control de Límites y Respuestas

### Límite de 4096 caracteres

Telegram rechaza mensajes de más de **4096 caracteres**. Toda respuesta debe fragmentarse:

```python
LIMITE_TELEGRAM: int = 4096

async def enviar_respuesta(message, texto: str) -> None:
    """Envía el texto fragmentándolo en bloques de máximo 4096 caracteres."""
    for i in range(0, len(texto), LIMITE_TELEGRAM):
        await message.reply_text(texto[i:i + LIMITE_TELEGRAM])
```

- Fragmenta preferentemente en saltos de línea o párrafos para no cortar oraciones a la mitad.
- Añade un pequeño `await asyncio.sleep(...)` entre bloques si el bot envía varios seguidos, para evitar límites de rate.

### Estados de usuario y banderas de sesión

Para estados simples (modo actual activado, confirmación pendiente, flujo multi-paso), usa `context.user_data` sin necesidad de bases de datos:

```python
async def activar_modo(update, context) -> None:
    """Activa una bandera de sesión para el usuario actual."""
    context.user_data["modo_activo"] = True
    await update.message.reply_text("Modo activado.")

async def handle_message(update, context) -> None:
    """Comprueba la bandera de sesión antes de procesar el mensaje."""
    if context.user_data.get("modo_activo"):
        # Procesar solo si la bandera está activa
        ...
```

- `context.user_data` es un dict por usuario y por chat, gestionado por la librería: ideal para estados temporales.
- Para estados persistentes entre reinicios, delega en `src/database`, nunca en el bot.

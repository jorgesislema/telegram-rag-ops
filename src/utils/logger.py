# -*- coding: utf-8 -*-
"""Configuración centralizada de logging para todo el proyecto.

Expone la función `get_logger` que devuelve un logger configurado con
doble salida (consola y archivo rotativo). Todos los módulos del sistema
deben consumir este logger en lugar de usar `print()`.

Salidas configuradas:
- Consola (StreamHandler): nivel INFO.
- Archivo rotativo `logs/app.log` (RotatingFileHandler): nivel DEBUG,
  máximo 5 MB por archivo y 3 respaldos rotativos.
"""

import logging
import logging.handlers
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, TextIO

# Ruta base del proyecto (src/utils/logger.py -> subir dos niveles)
_RUTA_BASE: Final[Path] = Path(__file__).resolve().parent.parent.parent
# Directorio de logs relativo a la raíz del proyecto
DIRECTORIO_LOGS: Path = _RUTA_BASE / "logs"
ARCHIVO_LOG: Path = DIRECTORIO_LOGS / "app.log"

# Formato uniforme para todos los handlers
FORMATO_LOG: Final[str] = (
    "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
)

# Parámetros de rotación del archivo de log
TAMANO_MAXIMO_BYTES: Final[int] = 5 * 1024 * 1024  # 5 MB
NUMERO_RESPALDOS: Final[int] = 3


@dataclass
class EstadoLogger:
    """Estado mutable del módulo de logging.

    Agrupa en una sola clase los valores que cambian en tiempo de
    ejecución (bandera de configuración y handlers instalados) para
    poder reemplazarlos con facilidad desde las pruebas.
    """

    # Bandera que indica si los handlers ya fueron instalados (evita
    # duplicados incluso si terceros, como pytest, configuran logging antes)
    configurado: bool = False
    # Referencias a los handlers instalados por este módulo. Permite
    # identificarlos aunque terceros agreguen handlers propios al raíz.
    handlers_instalados: list[logging.Handler] = field(default_factory=lambda: [])


# Estado único del módulo (reemplazable en pruebas con objeto nuevo)
estado: EstadoLogger = EstadoLogger()


def _configurar_handlers() -> None:
    """Crea la carpeta de logs e instala los handlers en el logger raíz.

    Idempotente en la práctica: `get_logger` garantiza que solo se invoque
    una vez por vida del proceso mediante la bandera `configurado`.
    """
    # Crear la carpeta logs/ automáticamente si no existe
    DIRECTORIO_LOGS.mkdir(parents=True, exist_ok=True)

    # Formateador compartido por ambos handlers
    formateador: logging.Formatter = logging.Formatter(FORMATO_LOG)

    # Logger raíz: punto único de configuración para toda la aplicación
    logger_raiz: logging.Logger = logging.getLogger()
    logger_raiz.setLevel(logging.DEBUG)

    # Salida por consola: solo eventos INFO o superiores
    handler_consola: logging.StreamHandler[TextIO] = logging.StreamHandler(sys.stderr)
    handler_consola.setLevel(logging.INFO)
    handler_consola.setFormatter(formateador)
    logger_raiz.addHandler(handler_consola)

    # Salida a archivo rotativo: captura eventos DEBUG o superiores
    handler_archivo: logging.handlers.RotatingFileHandler = (
        logging.handlers.RotatingFileHandler(
            filename=ARCHIVO_LOG,
            maxBytes=TAMANO_MAXIMO_BYTES,
            backupCount=NUMERO_RESPALDOS,
            encoding="utf-8",
        )
    )
    handler_archivo.setLevel(logging.DEBUG)
    handler_archivo.setFormatter(formateador)
    logger_raiz.addHandler(handler_archivo)

    # Registrar los handlers instalados para poder identificarlos después
    estado.handlers_instalados = [handler_consola, handler_archivo]


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger configurado con el nombre indicado.

    Args:
        name: Nombre del logger, habitualmente `__name__` del módulo
            que lo solicita.

    Returns:
        Un `logging.Logger` con doble salida (consola en INFO y
        archivo rotativo en DEBUG) ya configurada.
    """
    # Bandera explícita en lugar de inspeccionar los handlers existentes:
    # librerías de terceros (o pytest) pueden instalar handlers propios
    # antes de la primera llamada, lo que bloquearía nuestra configuración.
    if not estado.configurado:
        _configurar_handlers()
        # Marcar como configurado para no repetir la instalación
        estado.configurado = True

    return logging.getLogger(name)

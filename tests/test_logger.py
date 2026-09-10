# -*- coding: utf-8 -*-
"""Pruebas unitarias del módulo `src/utils/logger.py`.

Cobertura de:
- `get_logger()` devuelve una instancia válida de `logging.Logger`.
- La carpeta `logs/` y el archivo `logs/app.log` se generan automáticamente.
- El formateador escribe los mensajes correctamente en consola.
- Las excepciones se capturan con su traza completa (stack trace).

Los tests son totalmente independientes: usan `tmp_path` para aislar el
directorio de logs y fixtures que restauran el estado global de
`logging` después de cada prueba, evitando interferencias.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Generator, List

import pytest  # type: ignore[import-not-found]

from src.utils import logger as modulo_logger
from src.utils.logger import get_logger


@pytest.fixture(autouse=True)
def aislar_logging(tmp_path: Path) -> Generator[Path, None, None]:
    """Aísla el logging global y redirige los archivos de log a tmp_path.

    Restaura el estado original de `logging` (handlers y nivel del logger
    raíz) al terminar cada prueba para no interferir con otros tests.

    Args:
        tmp_path: Directorio temporal único por prueba (fixture de pytest).

    Yields:
        La ruta temporal donde se escribirán los logs de la prueba.
    """
    # Redirigir el directorio de logs hacia tmp_path para no tocar logs/ real
    modulo_logger.DIRECTORIO_LOGS = tmp_path / "logs"
    modulo_logger.ARCHIVO_LOG = modulo_logger.DIRECTORIO_LOGS / "app.log"

    # Guardar el estado original del módulo y del logger raíz
    logger_raiz: logging.Logger = logging.getLogger()
    handlers_originales: List[logging.Handler] = list(logger_raiz.handlers)
    nivel_original: int = logger_raiz.level
    estado_original: modulo_logger.EstadoLogger = modulo_logger.estado
    # Sustituir el estado por uno fresco para forzar la reconfiguración
    modulo_logger.estado = modulo_logger.EstadoLogger()
    logger_raiz.handlers = []
    yield tmp_path

    # Cerrar los handlers de archivo abiertos durante la prueba
    for handler in logger_raiz.handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            handler.close()
    # Restaurar el estado original tras la prueba (limpieza total)
    logger_raiz.handlers = handlers_originales
    logger_raiz.setLevel(nivel_original)
    modulo_logger.estado = estado_original


def test_get_logger_retorna_instancia_valida() -> None:
    """Verifica que `get_logger` devuelve un `logging.Logger` válido."""
    logger: logging.Logger = get_logger("test.instancia")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.instancia"


def test_get_logger_configura_doble_salida() -> None:
    """Verifica que se instalan dos handlers: consola y archivo rotativo."""
    get_logger("test.handlers")

    handlers: List[logging.Handler] = modulo_logger.estado.handlers_instalados
    tipos: List[type] = [type(h) for h in handlers]
    assert logging.StreamHandler in tipos
    assert logging.handlers.RotatingFileHandler in tipos


def test_get_logger_niveles_de_handlers() -> None:
    """Verifica que la consola usa INFO y el archivo rotativo usa DEBUG."""
    get_logger("test.niveles")

    # Usar los handlers instalados por el módulo: pytest agrega handlers
    # propios al logger raíz que contaminarían la verificación.
    # OJO: RotatingFileHandler hereda de StreamHandler, se comprueba primero.
    for handler in modulo_logger.estado.handlers_instalados:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            # El handler de archivo captura desde DEBUG
            assert handler.level == logging.DEBUG
        elif isinstance(handler, logging.StreamHandler):
            # El handler de consola solo emite INFO o superior
            assert handler.level == logging.INFO


def test_crea_carpeta_y_archivo_de_logs() -> None:
    """Verifica que `logs/` y `logs/app.log` se generan automáticamente."""
    logger: logging.Logger = get_logger("test.archivos")

    # Escribir un registro para forzar la creación del archivo en disco
    logger.info("mensaje para forzar la creacion del archivo")

    assert modulo_logger.DIRECTORIO_LOGS.is_dir()
    assert modulo_logger.ARCHIVO_LOG.is_file()


def test_formato_de_mensaje_en_archivo() -> None:
    """Verifica que el archivo respeta el formato `asctime | nivel | ...`."""
    logger: logging.Logger = get_logger("test.formato")
    logger.info("mensaje de formato")

    # Vaciar buffers del handler de archivo antes de leerlo en disco
    for handler in logging.getLogger().handlers:
        handler.flush()

    contenido: str = modulo_logger.ARCHIVO_LOG.read_text(encoding="utf-8")
    # El formato debe incluir nivel, nombre del logger y el propio mensaje
    assert "INFO" in contenido
    assert "test.formato" in contenido
    assert "mensaje de formato" in contenido
    # Separadores del formato definido en logger.py
    assert " | " in contenido


def test_salida_por_consola(capsys: pytest.CaptureFixture[str]) -> None:
    """Verifica que los mensajes INFO llegan a la consola (stderr).

    Args:
        capsys: Fixture de pytest para capturar stdout/stderr.
    """
    logger: logging.Logger = get_logger("test.consola")
    logger.info("mensaje visible en consola")

    # StreamHandler escribe sobre sys.stderr por defecto
    capturado = capsys.readouterr()
    assert "mensaje visible en consola" in capturado.err
    # Los DEBUG no deben aparecer en consola (nivel INFO en StreamHandler)
    logger.debug("mensaje debug invisible")
    capturado = capsys.readouterr()
    assert "mensaje debug invisible" not in capturado.err


def test_debug_solo_en_archivo() -> None:
    """Verifica que los mensajes DEBUG se escriben solo en el archivo."""
    logger: logging.Logger = get_logger("test.debug")
    logger.debug("mensaje solo archivo")

    for handler in logging.getLogger().handlers:
        handler.flush()

    contenido: str = modulo_logger.ARCHIVO_LOG.read_text(encoding="utf-8")
    assert "mensaje solo archivo" in contenido


def test_excepcion_con_traza_completa() -> None:
    """Verifica que `logger.exception` registra mensaje y stack trace."""
    logger: logging.Logger = get_logger("test.excepcion")

    try:
        # Provocar una excepción controlada para capturar su traza
        raise ValueError("error provocado para la prueba")
    except ValueError:
        # exception() adjunta la traza completa del error en curso
        logger.exception("fallo controlado")

    for handler in logging.getLogger().handlers:
        handler.flush()

    contenido: str = modulo_logger.ARCHIVO_LOG.read_text(encoding="utf-8")
    # El mensaje de error y la traza (tipo y archivo) deben estar presentes
    assert "fallo controlado" in contenido
    assert "ValueError" in contenido
    assert "error provocado para la prueba" in contenido
    assert "test_logger.py" in contenido


def test_get_logger_no_duplica_handlers() -> None:
    """Verifica que llamadas repetidas no añaden handlers duplicados."""
    get_logger("test.duplicado")
    total_primera: int = len(modulo_logger.estado.handlers_instalados)

    # Llamadas posteriores no deben reconfigurar ni duplicar handlers
    get_logger("test.duplicado.otra")
    total_segunda: int = len(modulo_logger.estado.handlers_instalados)

    assert total_primera == total_segunda
    # Exactamente dos handlers propios: consola y archivo rotativo
    assert total_primera == 2


def test_parametros_de_rotacion() -> None:
    """Verifica los parámetros del RotatingFileHandler (5 MB y 3 respaldos)."""
    get_logger("test.rotacion")
    for handler in modulo_logger.estado.handlers_instalados:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            # 5 MB por archivo y 3 respaldos rotativos
            assert handler.maxBytes == 5 * 1024 * 1024
            assert handler.backupCount == 3
            break

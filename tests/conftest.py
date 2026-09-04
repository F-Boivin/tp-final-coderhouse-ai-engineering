"""Fixtures compartidas de la suite.

Todo lo que la suite necesita se construye en memoria: los dobles de `tests/dobles.py`, un
`InMemorySaver` y un corpus sintético. La suite no toca Redis, ni Docker, ni la API de ningún
proveedor, así que corre igual en una máquina sin claves.
"""

import pytest

from app.grafo.estado import (
    Aprobacion,
    Investigacion,
    Redaccion,
    Verificacion,
    huella_material,
    huella_texto,
)
from tests.dobles import (
    FALLOS,
    SUBSECCION,
    RedisFalso,
    citas,
    estado_inicial,
    pasadas,
)


@pytest.fixture
def redis_falso() -> RedisFalso:
    """Un Redis en memoria, limpio para cada test."""
    return RedisFalso()


@pytest.fixture
def estado_vacio() -> dict:
    """El estado con el que arranca cualquier trabajo."""
    return estado_inicial()


@pytest.fixture
def investigacion() -> Investigacion:
    """Una investigación con tres citas del padrón de prueba."""
    return Investigacion(
        sintesis="Sintesis de prueba sobre el exceso ritual manifiesto.",
        citas=citas(3),
        subsecciones=(SUBSECCION,),
    )


@pytest.fixture
def verificacion(investigacion: Investigacion) -> Verificacion:
    """El veredicto que aprueba las tres citas de esa investigación."""
    return Verificacion(
        verificadas=tuple(c.fallo for c in investigacion.citas),
        inexistentes=(),
        aprobado=True,
        observaciones=(),
    )


@pytest.fixture
def redaccion(investigacion: Investigacion, verificacion: Verificacion) -> Redaccion:
    """Una redacción limpia escrita sobre ese material."""
    usadas = verificacion.verificadas
    return Redaccion(
        texto=f"Respuesta apoyada en {', '.join(usadas)}.",
        citas_usadas=usadas,
        citas_intrusas=(),
        motivos=(),
        limpia=True,
        llamadas=1,
        sobre_material=huella_material(investigacion, verificacion),
    )


@pytest.fixture
def estado_terminado(estado_vacio, investigacion, verificacion, redaccion) -> dict:
    """El estado de un trabajo con las tres etapas hechas y nada pendiente."""
    return {
        **estado_vacio,
        "investigaciones": (investigacion,),
        "verificaciones": (verificacion,),
        "redacciones": (redaccion,),
    }


@pytest.fixture
def aprobacion(redaccion: Redaccion):
    """Fabrica un veredicto humano sobre el texto de esa redacción."""
    def construir(accion: str, motivos=()) -> Aprobacion:
        return Aprobacion(
            accion=accion,
            revisor="felipe",
            motivos=tuple(motivos),
            sobre_texto=huella_texto(redaccion.texto),
            decidida_en="2026-09-01T12:00:00+00:00",
        )
    return construir


@pytest.fixture(autouse=True)
def contador_limpio():
    """Resetea el contador de pasadas del nodo de publicación entre tests."""
    pasadas["antes"] = 0
    pasadas["despues"] = 0
    yield


@pytest.fixture
def fallos() -> tuple:
    """Los fallos del padrón de prueba."""
    return FALLOS

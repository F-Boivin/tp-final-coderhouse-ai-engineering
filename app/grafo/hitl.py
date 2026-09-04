"""El nodo que publica, y que se detiene cuando el trabajo necesita una persona.

La consigna pide una pausa obligatoria "en tareas que el sistema identifique como críticas".
Este nodo hace las dos cosas: identifica y, si corresponde, se detiene.

El criterio es la calidad del propio trabajo, medida en `evaluar_calidad`. Un resultado con
todas sus citas verificadas, apoyado en material holgado y sin haber agotado las correcciones
se publica solo. Uno que falló alguno de los tres umbrales espera a un revisor, y le muestra
exactamente qué lo hizo dudar.

El revisor tiene tres respuestas: publicar, ampliar la búsqueda o rechazar el texto.
"""

from langgraph.types import interrupt
from pydantic import ValidationError

import app.nucleo.constantes as cfg
from app.nucleo.errores import ErrorDeAgente
from app.grafo.estado import (
    ACCIONES,
    APROBACION,
    Aprobacion,
    Calidad,
    EstadoOrquestador,
    evaluar_calidad,
    huella_texto,
    leer_situacion,
)


def calidad_del_trabajo(state: EstadoOrquestador) -> Calidad:
    """La medición del trabajo, con los umbrales de configuración ya aplicados."""
    return evaluar_calidad(
        state,
        holgadas=cfg.CITAS_HOLGADAS,
        tope_intentos=cfg.MAXIMO_INTENTOS,
    )


def pedido_de_aprobacion(state: EstadoOrquestador, calidad: Calidad) -> dict:
    """Lo que ve el revisor para decidir.

    Lleva los motivos que dispararon la pausa: el revisor sabe qué mirar antes de leer el
    texto entero.

    Se calcula solo del estado, sin reloj ni azar ni efectos: las dos ejecuciones del nodo
    —la que corta y la que reanuda— producen el mismo diccionario.
    """
    situacion = leer_situacion(state)
    redaccion = situacion.redaccion
    verificadas = set(situacion.verificacion.verificadas)
    return {
        "consulta": state["consulta"],
        "texto": redaccion.texto,
        "citas": [
            {"fallo": cita.fallo, "afirmacion": cita.afirmacion}
            for cita in situacion.investigacion.citas
            if cita.fallo in verificadas
        ],
        "sobre_texto": huella_texto(redaccion.texto),
        "intentos": state.get("intentos", 0),
        "motivos": list(calidad.motivos),
        "cobertura": round(calidad.cobertura, 3),
        "citas_en_el_texto": calidad.citas_en_el_texto,
    }


def publicar_sin_revision(calidad: Calidad) -> dict:
    """Cierra el trabajo cuando ninguna señal pidió intervención."""
    return {
        "publicado": True,
        "messages": [(
            "assistant",
            f"[{APROBACION}] {cfg.MOTIVO_SIN_REVISION} · cobertura "
            f"{calidad.cobertura:.0%}, {calidad.citas_en_el_texto} citas en el texto",
        )],
    }


def leer_veredicto(respuesta: dict, sobre_texto: str) -> Aprobacion:
    """Convierte lo que mandó el revisor en el artefacto que va al estado."""
    accion = str((respuesta or {}).get("accion", "")).strip().lower()
    if accion not in ACCIONES:
        raise ErrorDeAgente(cfg.ERROR_ACCION_INVALIDA.format(accion=accion))
    try:
        return Aprobacion(
            accion=accion,
            revisor=str(respuesta["revisor"]),
            motivos=tuple(respuesta.get("motivos") or ()),
            # Del estado y no del cuerpo que mandó el revisor: una decisión no puede declarar
            # que es sobre otro texto.
            sobre_texto=sobre_texto,
            decidida_en=str(respuesta["decidida_en"]),
        )
    except (ValidationError, KeyError, TypeError) as exc:
        raise ErrorDeAgente(cfg.ERROR_APROBACION_INVALIDA.format(detalle=exc)) from exc


async def aprobacion_node(state: EstadoOrquestador) -> dict:
    """Publica el trabajo, o lo detiene hasta que un humano decida qué hacer con él.

    LangGraph re-ejecuta el nodo desde el principio al reanudar. Todo lo anterior al
    `interrupt()` es puro; nada con efecto puede ir ahí arriba porque correría dos veces.
    """
    calidad = calidad_del_trabajo(state)   # puro: idéntico en las dos pasadas
    if not calidad.critico:
        return publicar_sin_revision(calidad)

    pedido = pedido_de_aprobacion(state, calidad)   # puro
    respuesta = interrupt(pedido)                   # frontera: corta acá, y vuelve acá
    # Debajo corre una sola vez, con la decisión ya tomada.
    veredicto = leer_veredicto(respuesta, pedido["sobre_texto"])

    firma = (veredicto.accion if veredicto.aprobado
             else f"{veredicto.accion}: " + " | ".join(veredicto.motivos))
    return {
        "aprobaciones": (veredicto,),
        "publicado": veredicto.aprobado,
        "messages": [("assistant", f"[{APROBACION}] {firma} — por {veredicto.revisor}")],
    }

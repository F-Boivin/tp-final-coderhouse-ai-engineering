"""Verifica el ciclo de publicación sin API, sin Redis y sin gastar en el modelo.

Reemplaza los tres especialistas por dobles deterministas y usa `InMemorySaver`, así cada
comprobación tarda milisegundos y se puede repetir. Lo que NO se reemplaza es el nodo de
publicación ni el ruteo: son las piezas bajo prueba.

Los dobles viven en `tests/dobles.py`, donde también los usa la suite de pytest.

Uso:
    python verificar_hitl.py
"""

import asyncio
import sys

from langgraph.types import Command

import app.nucleo.constantes as cfg
from app.grafo.estado import AMPLIAR, PUBLICAR, RECHAZAR, huella_texto, leer_situacion
from tests.dobles import (
    armar,
    estado_inicial,
    pasadas,
    redactor_escaso,
    redactor_sucio,
    supervisor_con_freno,
    veredicto,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def config(hilo: str, limite: int = 40) -> dict:
    """El config de una corrida."""
    return {"configurable": {"thread_id": hilo}, "recursion_limit": limite}


def separador(titulo: str) -> None:
    """Encabezado de bloque."""
    print("\n" + "-" * 86)
    print(titulo)
    print("-" * 86)



async def bloque_limpio() -> bool:
    """[1] Un trabajo sin señales de alerta se publica sin molestar a nadie."""
    separador("[1] EL TRABAJO LIMPIO SE PUBLICA SOLO — sin pausa")
    grafo = armar()
    salida = await grafo.ainvoke(estado_inicial(), config("t1"))

    interrupciones = salida.get("__interrupt__") or ()
    redaccion = salida["redacciones"][-1]
    print(f"    citas en el texto : {len(redaccion.citas_usadas)} "
          f"(umbral de holgura: {cfg.CITAS_HOLGADAS})")
    print(f"    intentos          : {salida['intentos']} de {cfg.MAXIMO_INTENTOS}")
    print(f"    interrupciones    : {len(interrupciones)}")
    print(f"    publicado         : {salida['publicado']}")
    print(f"    aprobaciones      : {len(salida['aprobaciones'])} (nadie tuvo que intervenir)")
    return not interrupciones and salida["publicado"] and not salida["aprobaciones"]


async def bloque_pausa() -> bool:
    """[2] Un trabajo flojo se detiene y dice exactamente qué lo hizo dudar."""
    separador("[2] EL TRABAJO FLOJO SE DETIENE — y muestra por qué")
    grafo = armar(redactor=redactor_escaso)
    salida = await grafo.ainvoke(estado_inicial(), config("t2"))

    interrupciones = salida.get("__interrupt__") or ()
    if not interrupciones:
        print("    NO pausó: el criterio de calidad no detectó nada")
        return False
    pedido = interrupciones[0].value
    print(f"    pausó con interrupt_id {interrupciones[0].id}")
    print(f"    motivos            : {pedido['motivos']}")
    print(f"    cobertura          : {pedido['cobertura']:.0%}")
    print(f"    citas en el texto  : {pedido['citas_en_el_texto']}")
    print(f"    huella del texto   : {pedido['sobre_texto']}")
    return bool(pedido["motivos"]) and pedido["citas_en_el_texto"] < cfg.CITAS_HOLGADAS


async def bloque_publicar() -> bool:
    """[3] Publicar cierra el grafo y deja el veredicto en el estado."""
    separador("[3] PUBLICAR — el revisor autoriza y el grafo cierra")
    grafo = armar(redactor=redactor_escaso)
    await grafo.ainvoke(estado_inicial(), config("t3"))
    final = await grafo.ainvoke(Command(resume=veredicto(PUBLICAR)), config("t3"))

    aprobacion = final["aprobaciones"][-1]
    print(f"    publicado : {final['publicado']}  ·  revisor: {aprobacion.revisor}  ·  "
          f"accion: {aprobacion.accion}")
    print(f"    la huella aprobada coincide con el texto: "
          f"{aprobacion.sobre_texto == huella_texto(final['redacciones'][-1].texto)}")
    return final["publicado"] and len(final["aprobaciones"]) == 1


async def bloque_rechazar() -> bool:
    """[4] Rechazar manda a reescribir el texto, con el mismo material."""
    separador("[4] RECHAZAR — vuelve al redactor")
    grafo = armar(redactor=redactor_escaso)
    await grafo.ainvoke(estado_inicial(), config("t4"))
    segunda = await grafo.ainvoke(
        Command(resume=veredicto(RECHAZAR, ["Falta citar el precedente completo."])),
        config("t4"),
    )

    investigaciones = len(segunda["investigaciones"])
    redacciones = len(segunda["redacciones"])
    print(f"    investigaciones : {investigaciones} (el material no cambia)")
    print(f"    redacciones     : {redacciones} (el texto sí)")
    print(f"    intentos        : {segunda['intentos']}")
    interrupciones = segunda.get("__interrupt__") or ()
    print(f"    volvió a pausar : {bool(interrupciones)}")
    return investigaciones == 1 and redacciones == 2 and segunda["intentos"] >= 1


async def bloque_ampliar() -> bool:
    """[5] Ampliar manda al investigador a buscar más, y eso cambia el material."""
    separador("[5] AMPLIAR — vuelve al investigador y trae más doctrina")
    grafo = armar(redactor=redactor_escaso)
    primera = await grafo.ainvoke(estado_inicial(), config("t5"))
    pedido = (primera.get("__interrupt__") or (None,))[0]
    if pedido is None:
        print("    no pausó: no hay nada que ampliar")
        return False
    citas_antes = len(primera["investigaciones"][-1].citas)
    print(f"    citas antes de ampliar : {citas_antes}")

    final = await grafo.ainvoke(
        Command(resume=veredicto(AMPLIAR, ["Solo cita una subseccion; falta doctrina."])),
        config("t5"),
    )
    citas_despues = len(final["investigaciones"][-1].citas)
    print(f"    investigaciones        : {len(final['investigaciones'])} (el material cambió)")
    print(f"    citas después          : {citas_despues}")
    print(f"    intentos consumidos    : {final['intentos']}")
    print(f"    redacciones            : {len(final['redacciones'])} (se reescribió sobre el material nuevo)")
    print(f"    accion registrada      : {final['aprobaciones'][-1].accion}")
    print(f"    publicado              : {final['publicado']}")
    print("    Con más material la respuesta dejó de ser crítica y el sistema publicó sin")
    print("    volver a molestar. Ampliar es un problema del material y lo arregla el")
    print("    investigador; rechazar es del texto y lo arregla el redactor.")
    return (len(final["investigaciones"]) == 2 and citas_despues > citas_antes
            and final["aprobaciones"][-1].accion == AMPLIAR
            and len(final["redacciones"]) == 2 and final["publicado"])


async def bloque_freno() -> bool:
    """[6] Un cierre por freno NO pausa: nadie queda esperando a un humano."""
    separador("[6] EL CIERRE POR FRENO NO PAUSA — nadie queda esperando")
    grafo = armar(redactor=redactor_sucio, supervisor=supervisor_con_freno)
    final = await grafo.ainvoke(estado_inicial(), config("t6", limite=60))

    interrupciones = final.get("__interrupt__") or ()
    situacion = leer_situacion(final)
    print(f"    intentos consumidos     : {final['intentos']}  ·  listo: {situacion.listo}")
    print(f"    interrupciones pendientes: {len(interrupciones)}")
    print(f"    publicado               : {final['publicado']}")
    print("    Un trabajo que agotó los frenos cierra solo, sin pedir aprobación.")
    return not interrupciones and not situacion.listo and not final["publicado"]


async def bloque_idempotencia() -> bool:
    """[7] El tramo previo al interrupt corre dos veces; el posterior, una."""
    separador("[7] IDEMPOTENCIA — el nodo se re-ejecuta desde el principio al reanudar")
    pasadas["antes"] = pasadas["despues"] = 0
    grafo = armar(redactor=redactor_escaso, contar=True)
    await grafo.ainvoke(estado_inicial(), config("t7"))
    print(f"    tras la pausa       -> antes: {pasadas['antes']}  despues: {pasadas['despues']}")
    await grafo.ainvoke(Command(resume=veredicto(PUBLICAR)), config("t7"))
    print(f"    tras la reanudación -> antes: {pasadas['antes']}  despues: {pasadas['despues']}")
    print("    Por eso nada con efecto puede ir arriba del interrupt().")
    return pasadas["antes"] == 2 and pasadas["despues"] == 1


async def verificar() -> int:
    """Corre los siete bloques y devuelve el código de salida."""
    print("=" * 86)
    print("CICLO DE PUBLICACIÓN — verificación sin API, sin Redis y sin modelo")
    print("=" * 86)
    print(f"umbrales · citas mínimas {cfg.CITAS_MINIMAS} · holgura {cfg.CITAS_HOLGADAS} · "
          f"intentos {cfg.MAXIMO_INTENTOS}")
    resultados = {
        "[1] el trabajo limpio se publica solo": await bloque_limpio(),
        "[2] el trabajo flojo se detiene y dice por qué": await bloque_pausa(),
        "[3] publicar cierra el grafo": await bloque_publicar(),
        "[4] rechazar vuelve al redactor": await bloque_rechazar(),
        "[5] ampliar vuelve al investigador": await bloque_ampliar(),
        "[6] el cierre por freno no pausa": await bloque_freno(),
        "[7] la idempotencia del nodo": await bloque_idempotencia(),
    }
    separador("RESUMEN")
    for nombre, cumplio in resultados.items():
        print(f"    {'OK   ' if cumplio else 'FALLA'}  {nombre}")
    if not all(resultados.values()):
        print(f"\n{cfg.ERROR_HITL_INCOMPLETO}", file=sys.stderr)
        return 1
    print(f"\nLos {len(resultados)} bloques hicieron lo que afirman.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(asyncio.run(verificar()))

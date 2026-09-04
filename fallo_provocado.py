"""Provoca una cita inventada y muestra al verificador cazándola.

El verificador y el resto del grafo son los de producción, sin tocar. Lo único que cambia es
el investigador: en su primera pasada se le agrega un fallo que no existe en el corpus, y de
ahí en adelante trabaja normal. La corrida demuestra que la cita inventada no llega al texto:
el verificador la rechaza y el investigador vuelve a trabajar. Si tampoco logra citar bien, se
agotan los intentos y el trabajo cierra sin publicar.

En 30 corridas medidas durante el módulo 7 el modelo citó bien siempre: un criterio que
espera el error por casualidad no se puede demostrar.

Uso:
    python fallo_provocado.py
"""

import asyncio
import sys

from langgraph.checkpoint.memory import InMemorySaver

import app.nucleo.constantes as cfg
import app.rag.herramientas as herramientas
import app.observabilidad.langsmith as observabilidad
from app.agentes.research_agent import investigador_node
from app.grafo.construccion import crear_grafo
from app.grafo.estado import Cita, Investigacion
from app.nucleo.config import cargar_entorno
from app.rag.ingesta import asegurar_indice

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CONSULTA = "¿Que es el exceso ritual manifiesto como causal de arbitrariedad?"
FALLO_IMPOSIBLE = "Fallos: 999:9999"
SUBSECCION = "6.2.7 Exceso ritual manifiesto"

pasadas = {"investigador": 0}


async def investigador_con_cita_falsa(state):
    """El investigador real, con un fallo inexistente agregado en la primera pasada."""
    pasadas["investigador"] += 1
    salida = await investigador_node(state)
    if pasadas["investigador"] > 1:
        return salida

    original = salida["investigaciones"][0]
    intrusa = Cita(
        fallo=FALLO_IMPOSIBLE,
        subseccion=SUBSECCION,
        afirmacion="Este fallo no existe en el corpus y el verificador tiene que cazarlo.",
    )
    adulterada = Investigacion(
        sintesis=original.sintesis,
        citas=original.citas + (intrusa,),
        subsecciones=original.subsecciones,
    )
    print(f"    [inyectado] se agrega {FALLO_IMPOSIBLE} a las {len(original.citas)} citas reales")
    return {**salida, "investigaciones": (adulterada,)}


def separador(titulo: str) -> None:
    print("\n" + "-" * 86)
    print(titulo)
    print("-" * 86)


async def provocar() -> int:
    cargar_entorno()
    await asyncio.to_thread(observabilidad.instrumentar)
    herramientas.inicializar(asegurar_indice(verboso=False))

    separador("FALLO PROVOCADO — una cita inventada, y el sistema cazándola")
    print(f"    consulta: {CONSULTA}")
    print(f"    el fallo inyectado no está entre las {len(herramientas.padron_de_citas())} "
          f"citas del corpus")

    grafo = crear_grafo(investigador=investigador_con_cita_falsa, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "fallo-provocado"},
              "recursion_limit": cfg.LIMITE_RECURSION_ORQUESTADOR,
              "run_name": "fallo-provocado",
              "metadata": {"prueba": "cita-inventada"}}
    final = await grafo.ainvoke({"messages": [], "consulta": CONSULTA,
                                 "siguiente": "investigador", "investigaciones": (),
                                 "verificaciones": (), "redacciones": (), "aprobaciones": (),
                                 "intentos": 0, "vueltas": 0, "completado": False,
                                 "publicado": False}, config)
    await asyncio.to_thread(observabilidad.esperar_trazas)

    separador("EL CICLO DE CORRECCIÓN, VEREDICTO POR VEREDICTO")
    for numero, veredicto in enumerate(final["verificaciones"], start=1):
        estado = "APROBADO" if veredicto.aprobado else "RECHAZADO"
        print(f"    verificación {numero}: {estado} · {len(veredicto.verificadas)} verificadas, "
              f"{len(veredicto.inexistentes)} inexistentes")
        for observacion in veredicto.observaciones:
            print(f"        {observacion[:100]}")

    inventada_cazada = any(
        FALLO_IMPOSIBLE in v.inexistentes for v in final["verificaciones"])
    corregida = final["verificaciones"][-1].aprobado if final["verificaciones"] else False
    limpia = bool(final["redacciones"]) and final["redacciones"][-1].limpia
    sin_rastro = all(
        FALLO_IMPOSIBLE not in r.texto for r in final["redacciones"])

    separador("RESUMEN")
    print(f"    investigaciones          : {len(final['investigaciones'])} "
          f"(el investigador corrió {pasadas['investigador']} veces)")
    print(f"    la cita falsa fue cazada : {inventada_cazada}")
    print(f"    y después se corrigió    : {corregida}")
    print(f"    la redacción quedó limpia: {limpia}")
    print(f"    el fallo falso no aparece en ningún texto: {sin_rastro}")
    print(f"    intentos consumidos      : {final['intentos']} de {cfg.MAXIMO_INTENTOS}")
    print(f"    publicado                : {final['publicado']}")

    if not (inventada_cazada and sin_rastro):
        print(f"\n{cfg.ERROR_FALLO_NO_CAZADO}", file=sys.stderr)
        return 1
    print()
    if corregida:
        print("    La cita inventada no llegó a la respuesta: el verificador la rechazó")
        print("    y el investigador volvió a citar, esta vez bien.")
    else:
        print("    La cita inventada no llegó a la respuesta. El investigador tampoco")
        print("    acertó en los reintentos, así que el trabajo cerró sin publicar: el")
        print("    sistema prefiere no responder antes que citar lo que no pudo verificar.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(provocar()))

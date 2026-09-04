"""Corre el sistema entero con el proveedor alternativo.

El factory elige el cliente por configuración: una variable de entorno cambia `ChatOpenAI` por
`ChatAnthropic` y el resto del sistema no se entera. Se comprueba en dos pasos: primero
construye los clientes de los dos proveedores sin llamar a ningún modelo, y después corre el
grafo completo con Anthropic contra el corpus real.

La corrida con Anthropic es paga y cuesta unas diez veces más que la misma con `gpt-4o-mini`;
por eso OpenAI es el proveedor por defecto.

Uso:
    python verificar_proveedor.py
"""

import asyncio
import os
import sys

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

import app.nucleo.constantes as cfg
import app.observabilidad.langsmith as observabilidad
import app.rag.herramientas as herramientas
from app.grafo.construccion import crear_grafo
from app.nucleo.config import (
    Proveedor,
    cargar_entorno,
    obtener_ajustes,
    reiniciar_ajustes,
)
from app.nucleo.modelos import crear_chat, modelo_del_rol
from app.rag.ingesta import asegurar_indice

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CONSULTA = "¿Que es el exceso ritual manifiesto como causal de arbitrariedad?"
ROLES = ("supervisor", "investigador", "redactor")
CLIENTE_ESPERADO = {Proveedor.OPENAI: ChatOpenAI, Proveedor.ANTHROPIC: ChatAnthropic}


def separador(titulo: str) -> None:
    print()
    print("-" * 86)
    print(titulo)
    print("-" * 86)


def usar(proveedor: Proveedor):
    """Activa un proveedor y devuelve los ajustes ya releídos.

    Los ajustes están cacheados: sin `reiniciar_ajustes` la variable nueva no se lee.
    """
    os.environ["LLM_PROVIDER"] = proveedor.value
    reiniciar_ajustes()
    return obtener_ajustes()


def bloque_factory() -> bool:
    """[1] Cada proveedor construye su propio cliente, sin gastar una sola llamada."""
    separador("[1] EL FACTORY — un cliente por proveedor, elegido por configuración")
    correcto = True
    for proveedor in Proveedor:
        ajustes = usar(proveedor)
        for rol in ROLES:
            modelo = crear_chat(rol, ajustes)
            clase = type(modelo).__name__
            esperado = CLIENTE_ESPERADO[proveedor].__name__
            correcto = correcto and clase == esperado
            marca = "" if clase == esperado else "  <- esperaba " + esperado
            print("    {:10} {:13} -> {:15} {}{}".format(
                proveedor.value, rol, clase, modelo_del_rol(rol, ajustes), marca))
    print()
    print("    El sistema pide `BaseChatModel` y el factory decide cuál construir. Agregar un")
    print("    proveedor es sumar una clase y una entrada al diccionario.")
    return correcto


async def bloque_corrida() -> tuple[bool, dict]:
    """[2] El grafo de producción entero, corriendo con Anthropic."""
    separador("[2] EL MISMO GRAFO, CON claude-haiku-4-5")
    ajustes = usar(Proveedor.ANTHROPIC)
    modelos = {rol: modelo_del_rol(rol, ajustes) for rol in ROLES}
    print("    proveedor activo : {}".format(ajustes.llm_provider.value))
    print("    modelos          : {}".format(modelos))
    print("    embeddings       : {} (Anthropic no ofrece API de embeddings)".format(
        ajustes.modelo_embeddings))
    print("    consulta         : {}".format(CONSULTA))

    grafo = crear_grafo(checkpointer=InMemorySaver())
    config = {
        "configurable": {"thread_id": "proveedor-anthropic"},
        "recursion_limit": cfg.LIMITE_RECURSION_ORQUESTADOR,
        "run_name": "orquestador-anthropic",
        "metadata": {"proveedor": ajustes.llm_provider.value, "modelos": modelos,
                     "prueba": "proveedor-alternativo"},
    }
    final = await grafo.ainvoke(
        {"messages": [], "consulta": CONSULTA, "siguiente": "investigador",
         "investigaciones": (), "verificaciones": (), "redacciones": (), "aprobaciones": (),
         "intentos": 0, "vueltas": 0, "completado": False, "publicado": False},
        config)
    await asyncio.to_thread(observabilidad.esperar_trazas)

    verificadas = sum(len(v.verificadas) for v in final["verificaciones"])
    print()
    print("    investigaciones  : {}".format(len(final["investigaciones"])))
    print("    citas verificadas: {}".format(verificadas))
    print("    redacciones      : {}".format(len(final["redacciones"])))
    if final["redacciones"]:
        ultima = final["redacciones"][-1]
        print("    texto            : {}...".format(ultima.texto[:160]))
        print("    citas usadas     : {}".format(len(ultima.citas_usadas)))
    return bool(final["redacciones"]) and verificadas > 0, final


async def verificar() -> int:
    cargar_entorno()
    await asyncio.to_thread(observabilidad.instrumentar)
    herramientas.inicializar(asegurar_indice(verboso=False))

    resultados = {"[1] el factory construye cada proveedor": bloque_factory()}
    resultados["[2] el grafo entero con anthropic"], _ = await bloque_corrida()

    separador("RESUMEN")
    for nombre, bien in resultados.items():
        print("    {:6} {}".format("OK" if bien else "FALLA", nombre))
    if not all(resultados.values()):
        print()
        print(cfg.ERROR_PROVEEDOR_NO_VERIFICADO, file=sys.stderr)
        return 1
    print()
    print("    El grafo, los agentes y las herramientas son los mismos. Lo único que cambió")
    print("    es qué cliente construyó el factory.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(verificar()))

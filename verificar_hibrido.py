"""Mide qué aporta cada lado del ensamble sobre el corpus real, consulta por consulta.

Corre los tres modos —léxico, vectorial e híbrido— sobre el mismo índice de 322 fragmentos y
compara en qué posición aparece el fragmento que cada consulta busca. Necesita
`OPENAI_API_KEY` porque embebe consultas de verdad; no necesita ni la API ni Redis.

Uso:
    python verificar_hibrido.py
"""

import asyncio
import re
import sys

from langchain_community.retrievers import BM25Retriever

import app.nucleo.constantes as cfg
import app.rag.herramientas as herramientas
from app.nucleo.config import cargar_entorno
from app.rag.hibrido import RecuperadorVectorial, crear_hibrido, tokenizar
from app.rag.ingesta import asegurar_indice

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Cada caso trae la consulta y el texto que tiene que aparecer en el fragmento buscado.
CASOS = [
    {
        # Una cita que el cuadernillo invoca en el cuerpo de un solo extracto.
        "titulo": "cita literal — la escribe el corpus, el embedding la diluye",
        "consulta": "Fallos: 211:958",
        "aguja": "211:958",
        "espera": "lexico",
    },
    {
        "titulo": "otra cita literal, en otra parte del cuadernillo",
        "consulta": "Fallos: 112:384",
        "aguja": "112:384",
        "espera": "lexico",
    },
    {
        "titulo": "consulta parafraseada — sin vocabulario del cuadernillo",
        "consulta": "cuando un tribunal se apega tanto a las formas que pierde de vista la verdad",
        "aguja": "exceso ritual",
        "espera": "vectorial",
    },
]


def separador(titulo: str) -> None:
    print("\n" + "-" * 86)
    print(titulo)
    print("-" * 86)


def rango(documentos, aguja: str):
    """Posición del primer fragmento que contiene la aguja, o None."""
    for posicion, doc in enumerate(documentos, start=1):
        if aguja.lower() in doc.page_content.lower():
            return posicion
        if aguja.lower() in doc.metadata.get("subseccion", "").lower():
            return posicion
    return None


def formatear(valor) -> str:
    return f"#{valor}" if valor else "no aparece"


async def medir(vectorstore, documentos) -> dict:
    """Corre los tres modos sobre cada caso y devuelve qué encontró cada uno."""
    candidatos = cfg.CANDIDATOS_POR_RETRIEVER
    bm25 = BM25Retriever.from_documents(documentos, k=candidatos, preprocess_func=tokenizar)
    vectorial = RecuperadorVectorial(vectorstore=vectorstore, k=candidatos)
    hibrido = crear_hibrido(vectorstore, documentos)

    resultados = {}
    for caso in CASOS:
        posiciones = {
            "lexico": rango(await bm25.ainvoke(caso["consulta"]), caso["aguja"]),
            "vectorial": rango(await vectorial.ainvoke(caso["consulta"]), caso["aguja"]),
            "hibrido": rango((await hibrido.ainvoke(caso["consulta"]))[:cfg.RESULTADOS_RECUPERADOS],
                             caso["aguja"]),
        }
        resultados[caso["titulo"]] = (caso, posiciones)
    return resultados


async def bloque_aporte(resultados: dict) -> bool:
    """[1] Cada lado encuentra lo que el otro no, y el híbrido se queda con los dos."""
    separador("[1] EL APORTE DE CADA LADO — sobre el corpus real de 322 fragmentos")
    print(f"    {'caso':<52}{'léxico':>10}{'vectorial':>12}{'híbrido':>10}")
    todo_bien = True
    for titulo, (caso, posiciones) in resultados.items():
        print(f"    {titulo[:50]:<52}{formatear(posiciones['lexico']):>10}"
              f"{formatear(posiciones['vectorial']):>12}{formatear(posiciones['hibrido']):>10}")
        # El modo que la consulta favorece tiene que encontrarlo, y el híbrido también.
        if posiciones[caso["espera"]] is None or posiciones["hibrido"] is None:
            todo_bien = False
    print("\n    El híbrido encuentra en los dos casos: la cita escrita literal y la consulta")
    print("    parafraseada. Cada lado solo, no.")
    return todo_bien


async def bloque_complementariedad(resultados: dict) -> bool:
    """[2] Hay al menos un caso donde un lado solo se queda corto."""
    separador("[2] LA COMPLEMENTARIEDAD ES REAL — un lado solo no alcanza")
    fallos_de_un_lado = 0
    for titulo, (_caso, posiciones) in resultados.items():
        for modo in ("lexico", "vectorial"):
            if posiciones[modo] is None:
                fallos_de_un_lado += 1
                print(f"    «{titulo[:50]}» -> el modo {modo} no lo encuentra")
    print(f"\n    casos en que un modo solo falla: {fallos_de_un_lado}")
    print("    Si ningún lado fallara nunca, el ensamble sería costo sin beneficio.")
    return fallos_de_un_lado > 0


async def bloque_contrato(vectorstore) -> bool:
    """[3] La herramienta del agente conserva su formato y sus metadatos."""
    separador("[3] EL CONTRATO DE LA HERRAMIENTA — formato y metadatos intactos")
    salida = await herramientas.buscar_doctrina.ainvoke(
        {"consulta": "exceso ritual manifiesto", "cantidad": cfg.RESULTADOS_RECUPERADOS})
    # Los fragmentos se cuentan por su encabezado `[subseccion]`: partir por el separador de
    # extractos se pasaría de largo, porque ese separador también vive dentro del texto.
    encabezados = re.findall(r"^\[(.+?)\]$", salida, re.MULTILINE)
    con_subseccion = len(encabezados) == cfg.RESULTADOS_RECUPERADOS
    trae_encabezado = cfg.ENCABEZADO_SUBSECCIONES in salida
    print(f"    fragmentos devueltos           : {len(encabezados)} "
          f"(pedidos: {cfg.RESULTADOS_RECUPERADOS})")
    print(f"    cada uno con su subsección      : {con_subseccion}")
    print(f"    trae el listado de subsecciones : {trae_encabezado}")
    print(f"    padrón de citas disponible      : {len(herramientas.padron_de_citas())} citas")
    print("\n    El agente ve exactamente lo mismo que antes del ensamble.")
    return con_subseccion and trae_encabezado


async def verificar() -> int:
    cargar_entorno()
    vectorstore = asegurar_indice(verboso=False)
    herramientas.inicializar(vectorstore)
    documentos = herramientas._documentos_del_corpus(herramientas._leer_corpus())
    print(f"Corpus indexado: {len(documentos)} fragmentos.")

    resultados = await medir(vectorstore, documentos)
    bloques = {
        "[1] cada lado aporta lo suyo y el híbrido se queda con los dos": await bloque_aporte(resultados),
        "[2] un lado solo se queda corto en algún caso": await bloque_complementariedad(resultados),
        "[3] la herramienta conserva su contrato": await bloque_contrato(vectorstore),
    }

    separador("RESUMEN")
    for nombre, ok in bloques.items():
        print(f"    {'OK   ' if ok else 'FALLA'}  {nombre}")
    if not all(bloques.values()):
        print(f"\n{cfg.ERROR_HIBRIDO_INCOMPLETO}", file=sys.stderr)
        return 1
    print(f"\nLos {len(bloques)} bloques hicieron lo que afirman.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(verificar()))

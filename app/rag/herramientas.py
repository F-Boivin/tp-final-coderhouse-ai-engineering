"""Herramientas del orquestador sobre el índice Chroma del cuadernillo REX.

Diseño anti-alucinación heredado de la pre-entrega 3: **ninguna URL oficial sale de un modelo**,
todas salen de los metadatos. `buscar_doctrina` devuelve los fragmentos SIN links, así que las
URLs llegan por dos vías y ninguna pasa por un modelo: `fallos_citados` se las da al investigador
por subsección, y `link_oficial` al redactor, acotada a las citas que el verificador ya aprobó.

Cuatro herramientas y tres consumidores distintos: dos las usa el ReAct del investigador, una el
ReAct del redactor, y a `verificar_citas` la invoca directamente el código del nodo verificador,
que es por qué es la única con `handle_tool_error = False`.
"""

import asyncio
import json
import re
import unicodedata

from chromadb.errors import ChromaError
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.tools import ToolException, tool
from openai import APIConnectionError, APIError, RateLimitError
from pydantic import BaseModel, Field

import app.nucleo.constantes as cfg
from app.nucleo.config import obtener_ajustes
from app.rag.hibrido import crear_hibrido
from app.rag.ingesta import contar_tokens
from app.observabilidad.trazas import anotar_documentos, span_de_embedding, span_de_recuperacion

PATRON_TOMO_PAGINA = re.compile(r"(\d+):(\d+)")
NL = chr(10)

# Toda mención a fallos anclada en la palabra, con la lista de números que la sigue. El ancla
# es lo que separa una cita de un número cualquiera: sin ella, "la audiencia de las 14:30" se
# leería como el tomo 14 página 30.
PATRON_LISTA_FALLOS = re.compile(
    r"\bfallos?\b\s*:?\s*((?:\d+(?:\s*:\s*\d+)?)(?:\s*(?:[;,/]|y)\s*(?:\d+(?:\s*:\s*\d+)?))*)",
    re.IGNORECASE,
)
PATRON_UN_NUMERO = re.compile(r"\d+(?:\s*:\s*\d+)?")


def citas_del_texto(texto: str) -> set[str]:
    """Las citas que un texto invoca, normalizadas a "tomo:pagina".

    El cuadernillo cita en lista y **abrevia**: `(Fallos: 315:356; 326:2759 y 3334)` son tres
    citas, y la última es la página 3334 del tomo 326 — el tomo se sobreentiende del anterior.
    Buscar solo `tomo:pagina` daría por no citada esa tercera, que es justo la forma en que el
    corpus le enseña al modelo a escribir.

    El alcance llega hasta las citas con números: una referencia como "la doctrina de
    Colalillo" queda afuera. El prompt del redactor la prohíbe, y conviene decir hasta dónde
    llega el control.
    """
    encontradas: set[str] = set()
    for mencion in PATRON_LISTA_FALLOS.finditer(texto or ""):
        tomo = None
        for pieza in PATRON_UN_NUMERO.findall(mencion.group(1)):
            if ":" in pieza:
                tomo, pagina = (p.strip() for p in pieza.split(":", 1))
            else:
                # Página suelta: hereda el tomo de la cita anterior de la misma lista. Sin
                # tomo previo es un número, y no una cita.
                if tomo is None:
                    continue
                pagina = pieza
            encontradas.add(f"{tomo}:{pagina}")
    return encontradas


def contar_llamadas(mensajes) -> int:
    """Cuántas llamadas a herramienta hizo un agente en su ciclo interno.

    El ciclo ReAct queda adentro del nodo, así que sin este número el multi-paso solo se ve
    leyendo el código. Cuenta las llamadas pedidas y no los mensajes: un modelo puede pedir
    varias en un mismo turno, y eso es lo que muestra cómo razonó.
    """
    return sum(len(getattr(m, "tool_calls", None) or []) for m in mensajes)


def cuantas_citas(texto: str) -> int:
    """Cuántas citas distintas invoca un texto. Sirve para exigir que una cita sea una sola."""
    return len(PATRON_TOMO_PAGINA.findall(texto or ""))

_vectorstore: Chroma | None = None
_hibrido: EnsembleRetriever | None = None


def inicializar(vectorstore: Chroma) -> None:
    """Inyecta el vectorstore, llena los caches y arma el retriever hibrido.

    Una sola lectura del corpus alimenta las tres cosas: el padron de citas, el indice de
    subsecciones y los documentos que indexa BM25. Leer por separado abriria la puerta a que
    describan estados distintos de la misma base.

    Todo lo cacheado se descarta al cambiar de base: sobrevivir a un cambio la dejaria
    describiendo una que ya no es la inyectada.
    """
    global _vectorstore, _padron, _subsecciones, _hibrido
    _vectorstore = vectorstore
    _padron = None
    _subsecciones = None
    _hibrido = None
    # Se llenan acá y no cuando alguien los pida. Las dos lecturas son síncronas y bloquean el
    # event loop unos 100 ms: acá eso ocurre en el arranque del proceso, antes de que haya un
    # solo consumidor andando. Perezosas, se las comía el primer trabajo mientras los otros dos
    # esperaban.
    registros = _leer_corpus()
    _padron = _armar_padron(registros)
    _subsecciones = _armar_subsecciones(registros)
    _hibrido = crear_hibrido(vectorstore, _documentos_del_corpus(registros))
    # El primer `contar_tokens` construye el codificador de tiktoken, y puede bajarlo por
    # red. Que ocurra acá y no dentro de la primera busqueda.
    contar_tokens("")


def _leer_corpus() -> dict:
    """Los 322 fragmentos con su texto y sus metadatos, en una sola lectura."""
    with span_de_recuperacion("chroma_corpus_completo", "todos los fragmentos"):
        return _base().get(include=["metadatas", "documents"])


def _documentos_del_corpus(registros: dict) -> list[Document]:
    """Reconstruye los `Document` del indice para que BM25 vea lo mismo que el vectorial."""
    documentos = registros.get("documents") or []
    metadatos = registros.get("metadatas") or []
    return [
        Document(page_content=texto, metadata=dict(metadata))
        for texto, metadata in zip(documentos, metadatos)
    ]


def _hibrido_actual() -> EnsembleRetriever:
    """El retriever hibrido, o el aviso de que falto inicializar."""
    if _hibrido is None:
        raise RuntimeError(cfg.ERROR_HERRAMIENTA_NO_INICIALIZADA)
    return _hibrido


def _base() -> Chroma:
    """Devuelve el vectorstore inyectado o falla ruidoso si falta la inicialización."""
    if _vectorstore is None:
        raise RuntimeError(cfg.ERROR_HERRAMIENTA_NO_INICIALIZADA)
    return _vectorstore


class EntradaBusqueda(BaseModel):
    """Entrada de buscar_doctrina: Pydantic valida antes de tocar la base."""

    consulta: str = Field(
        min_length=3,
        max_length=cfg.LARGO_MAXIMO_CONSULTA,
        description="Tema o pregunta jurídica a buscar, en lenguaje natural.",
    )
    cantidad: int = Field(
        default=cfg.RESULTADOS_RECUPERADOS,
        ge=1,
        le=cfg.MAXIMO_RESULTADOS,  # tope defensivo: acota el contexto por llamada
        description="Cantidad máxima de fragmentos a devolver.",
    )


class EntradaFallos(BaseModel):
    """Entrada de fallos_citados."""

    subseccion: str = Field(
        min_length=3,
        max_length=200,
        description=(
            "Nombre de la subsección del cuadernillo, de los que buscar_doctrina lista al "
            "final. Se acepta con su numeración o sin ella: «6.2.7 Exceso ritual "
            "manifiesto» y «Exceso ritual manifiesto» llegan al mismo lugar."
        ),
    )


@tool(args_schema=EntradaBusqueda)
async def buscar_doctrina(consulta: str, cantidad: int = cfg.RESULTADOS_RECUPERADOS) -> str:
    """Busca doctrina de la CSJN sobre sentencias arbitrarias en el cuadernillo indexado.

    Usá esta herramienta cuando el usuario pregunte por conceptos, causales,
    procedencia o trámite del recurso extraordinario por sentencia arbitraria.
    Devuelve fragmentos de doctrina con su subsección entre corchetes; los
    fragmentos citan números de fallo pero NO traen links: para los links usá
    fallos_citados. Busca por significado y por palabra exacta a la vez, así que
    sirve tanto para un tema como para un número de fallo escrito literal.
    No sirve para otras ramas del derecho ni para hechos actuales.

    Args:
        consulta: tema o pregunta a buscar (3 a 500 caracteres).
        cantidad: máximo de fragmentos a devolver (1 a 8; por defecto 4).

    Returns:
        Fragmentos ordenados por similitud, cada uno encabezado por su subsección.

    Raises:
        ToolException: si la base vectorial o la API de embeddings fallan; el
        mensaje explica el problema para que el modelo pueda informarlo.
    """
    try:
        with span_de_recuperacion("hibrido_fusion", consulta, k=cantidad,
                                  candidatos=cfg.CANDIDATOS_POR_RETRIEVER) as span:
            fusionados = await _hibrido_actual().ainvoke(consulta)
            # El recorte se aplica sobre la lista ya fusionada: cada retriever aporta más
            # candidatos justamente para que la fusión tenga de dónde elegir.
            documentos = fusionados[:cantidad]
            anotar_documentos(span, documentos)
    except (RateLimitError, APIConnectionError, APIError, ChromaError, OSError) as exc:
        raise ToolException(cfg.ERROR_HERRAMIENTA_BASE.format(detalle=exc)) from exc

    if not documentos:
        return cfg.MENSAJE_SIN_RESULTADOS

    partes = []
    subsecciones = []
    for doc in documentos:
        subseccion = doc.metadata.get("subseccion", "")
        if subseccion and subseccion not in subsecciones:
            subsecciones.append(subseccion)
        texto = doc.page_content.strip()
        if len(texto) > cfg.LARGO_MAXIMO_FRAGMENTO:
            texto = texto[: cfg.LARGO_MAXIMO_FRAGMENTO] + "…"
        partes.append(f"[{subseccion}]\n{texto}")

    # El separador entre fragmentos también aparece dentro de ellos: el corpus separa sus
    # extractos con `---` y el splitter lo conserva. Los encabezados `[subseccion]` son la
    # marca inequívoca de dónde empieza cada fragmento; partir esta salida por el separador
    # cuenta de más. El modelo se orienta por los encabezados, así que queda documentado.
    listado = "\n---\n".join(partes)
    return f"{listado}\n\n{cfg.ENCABEZADO_SUBSECCIONES} {'; '.join(subsecciones)}"


def _clave_fallo(cita: str) -> tuple[int, int]:
    """Orden cronológico aproximado: los Fallos se citan como tomo:página."""
    coincidencia = PATRON_TOMO_PAGINA.search(cita)
    if not coincidencia:
        return (10**9, 10**9)  # lo no parseable va al final
    return (int(coincidencia.group(1)), int(coincidencia.group(2)))


@tool(args_schema=EntradaFallos)
async def fallos_citados(subseccion: str) -> str:
    """Lista los fallos citados en una subsección del cuadernillo, con su link oficial.

    Usá esta herramienta DESPUÉS de buscar_doctrina, cuando el usuario pida links,
    fuentes verificables o el detalle de los fallos de una subsección. Las
    referencias salen de los metadatos del índice, nunca del modelo: si un fallo
    no aparece acá, no tiene link verificable en el cuadernillo.

    Args:
        subseccion: nombre exacto de la subsección, tomado de la lista final de
            buscar_doctrina (si viene con corchetes, se toleran).

    Returns:
        Lista "Fallos: tomo:página — URL" en orden cronológico, o, si la subsección
        no existe, un aviso con los nombres válidos para elegir uno y reintentar.

    Raises:
        ToolException: si la base vectorial falla.
    """
    # El indice guarda los titulos con su numeracion y el modelo los escribe sin ella, asi que
    # la consulta se hace contra el nombre resuelto. Es la misma tolerancia que el verificador
    # aplica por su lado. Todo esto va adentro del try porque resolver el nombre lee los
    # metadatos, y una caida de la base ahi tiene que salir como ToolException igual que la
    # consulta de mas abajo.
    try:
        exacta = await asyncio.to_thread(resolver_subseccion, subseccion)
        if exacta is None:
            validas = await asyncio.to_thread(subsecciones_del_corpus)
            return cfg.MENSAJE_SIN_SUBSECCION.format(
                subseccion=subseccion.strip().strip("[]").strip(),
                validas="; ".join(validas),
            )
        with span_de_recuperacion("chroma_por_subseccion", exacta):
            registros = await asyncio.to_thread(
                _base().get, where={"subseccion": exacta}, include=["metadatas"]
            )
        metadatas = registros.get("metadatas") or []
        if not metadatas:
            validas = await asyncio.to_thread(subsecciones_del_corpus)
            return cfg.MENSAJE_SIN_SUBSECCION.format(
                subseccion=exacta, validas="; ".join(validas)
            )
    except (ChromaError, OSError, ValueError) as exc:
        raise ToolException(cfg.ERROR_HERRAMIENTA_BASE.format(detalle=exc)) from exc

    subseccion = exacta

    fallos: dict[str, str] = {}
    for metadata in metadatas:
        fallos.update(json.loads(metadata.get("citas_urls", "{}")))

    lineas = [
        f"- {cita} — {url}"
        for cita, url in sorted(fallos.items(), key=lambda par: _clave_fallo(par[0]))
    ]
    encabezado = cfg.ENCABEZADO_FALLOS.format(subseccion=subseccion)
    return f"{encabezado}\n" + "\n".join(lineas)


class EntradaVerificacion(BaseModel):
    """Entrada de verificar_citas."""

    citas: list[str] = Field(
        min_length=1,
        max_length=40,
        description='Citas a comprobar, como las escribio el investigador. Ej: "Fallos: 311:2437".',
    )


_padron: dict[str, str] | None = None


def padron_de_citas() -> dict[str, str]:
    """Todas las citas del corpus, normalizadas a "tomo:pagina" -> URL oficial (o "" sin link).

    Se arma una sola vez leyendo los 322 fragmentos, y junta las citas de **dos lugares**:

    - las lineas "Fuente:", que son las unicas que traen URL oficial;
    - **el cuerpo de la doctrina**, donde el cuadernillo cita entre parentesis.

    Las del cuerpo se suman porque estan en el cuadernillo igual: armar el padron solo con
    las de "Fuente:" hacia que el verificador rechazara citas legitimas. Medido antes del
    cambio: 17 de las 574 citas del corpus quedaban afuera, un 3% de falsos rechazos. Entran
    con URL vacia, y `link_oficial` ya tiene el camino "verificada pero sin link".

    La verificacion NO se hace por subseccion a proposito. El investigador escribe el
    nombre de la subseccion de memoria y lo parafrasea -"Caracterizacion" en vez de
    "6.1.2 Caracterizacion"-, asi que una busqueda por nombre exacto daria por inexistente
    un fallo que si esta. Un fallo pertenece al corpus o no; de que subseccion salio es
    otra pregunta, y se responde aparte.
    """
    global _padron
    if _padron is None:
        _padron = _armar_padron(_leer_corpus())
    return _padron


def _armar_padron(registros: dict) -> dict[str, str]:
    """El padron a partir de los registros ya leidos."""
    if True:
        # Primero las del cuerpo, sin URL.
        acumulado: dict[str, str] = {
            cita: ""
            for documento in registros.get("documents") or []
            for cita in citas_del_texto(documento)
        }
        # Y encima las de "Fuente:", que si la traen. Van segundas a proposito: cuando una
        # cita esta en los dos lados, tiene que ganar la que aporta el link.
        for metadata in registros.get("metadatas") or []:
            for cruda, url in json.loads(metadata.get("citas_urls", "{}")).items():
                clave = normalizar_cita(cruda)
                if clave:
                    acumulado[clave] = url
        _padron = acumulado
    return _padron


def normalizar_cita(cita: str) -> str:
    """Reduce una cita a "tomo:pagina", que es lo unico que la identifica.

    "Fallos: 311:2437", "Fallos 311:2437" y "311:2437" son la misma cita; comparar las
    cadenas crudas daria por distinto lo que es igual.
    """
    coincidencia = PATRON_TOMO_PAGINA.search(cita or "")
    return f"{coincidencia.group(1)}:{coincidencia.group(2)}" if coincidencia else ""


@tool(args_schema=EntradaVerificacion)
async def verificar_citas(citas: list[str]) -> str:
    """Comprueba, contra los metadatos del corpus, cuales de esas citas existen.

    Usala para auditar una investigacion antes de darla por buena. El veredicto es un hecho:
    una cita esta en los metadatos del cuadernillo o no esta.

    Args:
        citas: entre 1 y 40 citas tal como fueron escritas.

    Returns:
        Una linea por cita: "<cita> | EXISTE | <url>", "<cita> | EXISTE" cuando la cita esta
        en el corpus pero sin link registrado, o "<cita> | NO EXISTE".

    Raises:
        ToolException: si la base vectorial falla.
    """
    try:
        padron = await asyncio.to_thread(padron_de_citas)
    except (ChromaError, OSError, ValueError) as exc:
        raise ToolException(cfg.ERROR_HERRAMIENTA_BASE.format(detalle=exc)) from exc

    lineas = []
    for cita in citas:
        clave = normalizar_cita(cita)
        if clave and clave in padron:
            url = padron[clave]
            lineas.append(f"{cita} | EXISTE | {url}" if url else f"{cita} | EXISTE")
        else:
            lineas.append(f"{cita} | NO EXISTE")
    return NL.join(lineas)


def leer_veredicto(texto: str) -> dict[str, bool]:
    """Convierte la salida de verificar_citas en {cita: existe}.

    Devuelve la existencia y no la URL. Una cita del cuerpo de la doctrina existe y no tiene
    link registrado, asi que leer la URL como si fuera el veredicto la daria por inexistente.
    Los links los reparte `link_oficial`, que consulta el padron directo.
    """
    veredicto: dict[str, bool] = {}
    for linea in texto.splitlines():
        partes = [p.strip() for p in linea.split("|")]
        if len(partes) >= 2:
            veredicto[partes[0]] = partes[1] == "EXISTE"
    return veredicto


_subsecciones: list[str] | None = None


def subsecciones_del_corpus() -> list[str]:
    """Los nombres exactos de subseccion, para poder senialar cuando el modelo los inventa.

    Se cachea como el padron y por lo mismo: el verificador la llama una vez por cita, y sin
    cache cada llamada releia los metadatos de los 322 fragmentos. Medido: 4 citas eran 4
    lecturas completas de Chroma, 102 ms; con cache es una sola.
    """
    global _subsecciones
    if _subsecciones is None:
        _subsecciones = _armar_subsecciones(_leer_corpus())
    return _subsecciones


def _armar_subsecciones(registros: dict) -> list[str]:
    """Los nombres de subseccion a partir de los registros ya leidos."""
    return sorted(
        {m.get("subseccion", "") for m in registros.get("metadatas") or [] if m.get("subseccion")}
    )


def _clave_subseccion(nombre: str) -> str:
    """Reduce un nombre de subseccion a lo comparable: sin numeracion, tildes ni caja.

    El modelo escribe "Caracterizacion" donde el indice dice "6.1.2 Caracterizacion". Exigir
    el nombre exacto daria por inventada una subseccion que existe; ignorar el nombre por
    completo dejaria pasar una inventada de verdad.
    """
    limpio = re.sub(r"^[\d.]+\s*", "", (nombre or "").strip().strip("[]").strip())
    limpio = unicodedata.normalize("NFD", limpio)
    return "".join(c for c in limpio if unicodedata.category(c) != "Mn").lower()


def subseccion_existe(nombre: str) -> bool:
    """True si esa subseccion esta en el corpus, tolerando la numeracion y las tildes."""
    return _clave_subseccion(nombre) in {_clave_subseccion(s) for s in subsecciones_del_corpus()}


def resolver_subseccion(nombre: str) -> str | None:
    """El nombre tal cual figura en el indice, a partir del que escribio el modelo.

    Primero prueba igualdad exacta, que es lo que hacia el codigo anterior: sin ese paso, dos
    titulos que solo se distinguen por su numeracion dejarian de ser alcanzables incluso
    escribiendolos tal cual figuran. Recien despues resuelve por clave, y devuelve None si
    hay mas de un candidato, porque elegir seria adivinar.
    """
    limpio = (nombre or "").strip().strip("[]").strip()
    subsecciones = subsecciones_del_corpus()
    if limpio in subsecciones:
        return limpio
    clave = _clave_subseccion(limpio)
    candidatos = [s for s in subsecciones if _clave_subseccion(s) == clave]
    return candidatos[0] if len(candidatos) == 1 else None


class EntradaLink(BaseModel):
    """Entrada de link_oficial."""

    cita: str = Field(
        min_length=3,
        max_length=120,
        description='Un fallo de la lista de citas verificadas. Ej: "Fallos: 311:2437".',
    )


def crear_link_oficial(verificadas):
    """Arma la herramienta de links, acotada a las citas aprobadas de ESTA corrida.

    La lista permitida se captura en el cierre en vez de leerse del padron completo, y es la
    decision que hace que esta herramienta no debilite nada. Si aceptara cualquiera de las 574
    citas del corpus, el redactor podria pedir el link de un fallo real pero ajeno al tema,
    citarlo, y la guarda lo dejaria pasar porque existe. **Verificado no es lo mismo que
    pertinente**, y la herramienta que reparte links no puede ser la que borre esa diferencia.

    Devuelve una tool nueva por corrida. Es barato -no toca la base al construirse- y evita el
    estado global mutable que haria falta para inyectar la lista de otro modo.
    """
    permitidas = {normalizar_cita(c) for c in verificadas} - {""}

    @tool(args_schema=EntradaLink)
    async def link_oficial(cita: str) -> str:
        """Devuelve el link oficial de la CSJN para un fallo ya verificado.

        Usala para cada fallo que vayas a citar en la respuesta final, asi el lector puede
        abrirlo. El link sale de los metadatos del corpus, nunca de tu memoria: si esta
        herramienta no te lo da, ese fallo no tiene link verificable.

        No sirve para buscar doctrina ni para averiguar si un fallo existe: solo responde por
        las citas que el verificador ya aprobo en esta consulta.

        Args:
            cita: el fallo tal como figura en tu lista de citas verificadas.

        Returns:
            "<cita> - <url>", o un aviso claro si la cita no esta entre las aprobadas.

        Raises:
            ToolException: si la base vectorial falla.
        """
        clave = normalizar_cita(cita)
        if not clave:
            return cfg.MENSAJE_CITA_ILEGIBLE.format(cita=cita)
        if clave not in permitidas:
            return cfg.MENSAJE_CITA_NO_APROBADA.format(cita=cita)
        try:
            padron = await asyncio.to_thread(padron_de_citas)
        except (ChromaError, OSError, ValueError) as exc:
            raise ToolException(cfg.ERROR_HERRAMIENTA_BASE.format(detalle=exc)) from exc
        url = padron.get(clave)
        if not url:
            return cfg.MENSAJE_SIN_LINK.format(cita=cita)
        return f"{cita} - {url}"

    # La consume un ReAct: el error vuelve como observacion y el modelo reacciona al texto.
    link_oficial.handle_tool_error = True
    return link_oficial


# --- Configuracion de las tools, toda junta ---

# Las del investigador las consume un ReAct: el error vuelve como observación y el modelo
# reacciona al texto, así que no corta el grafo.
buscar_doctrina.handle_tool_error = True
fallos_citados.handle_tool_error = True

# La del verificador la consume código, que espera el formato exacto
# "cita | EXISTE | url". Con handle_tool_error, una caída de la base volvería como string de
# error, leer_veredicto no lo parsearía y toda cita quedaría marcada como inexistente: el
# sistema acusaría al investigador de inventar. Explícito en False aunque sea el default:
# acá la ToolException tiene que llegar al except de verificador_node y salir como
# ErrorDeAgente.
verificar_citas.handle_tool_error = False

# La de `link_oficial` se asigna dentro de `crear_link_oficial`, porque esa tool se arma una
# por corrida y no existe todavía cuando corre este bloque.

# Las dos primeras son del investigador, que las recibe como lista porque las consume un
# ReAct. `verificar_citas` no va en ninguna lista: al verificador no lo maneja un modelo que
# elija herramientas, la invoca directo el codigo del nodo.
HERRAMIENTAS_INVESTIGACION = [buscar_doctrina, fallos_citados]

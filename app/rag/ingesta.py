"""Ingesta: lee /data, fragmenta por extractos y persiste los vectores en ChromaDB.

El chunking respeta la unidad semantica del cuadernillo: cada extracto de doctrina con
su cita es indivisible, y el splitter solo lo parte si por si solo supera el techo de
tokens. La seccion y el subtitulo viajan como metadatos de cada fragmento.
"""

import json
from typing import Optional
import re
from pathlib import Path

import tiktoken
from chromadb.errors import ChromaError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import APIConnectionError, APIError, OpenAIError, RateLimitError

import app.nucleo.constantes as cfg
from app.nucleo.config import cargar_entorno, obtener_ajustes
from app.nucleo.errores import ErrorDeConfiguracion, ErrorRAG
from app.nucleo.modelos import crear_embeddings

PATRON_SUBTITULO = re.compile(r"^#{2,6}\s+(.*)$", re.MULTILINE)
PATRON_LINK_MD = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
PATRON_CITAS = re.compile(rf"^{cfg.ETIQUETA_FUENTE}\s*(.+)$", re.MULTILINE)
CLAVE_URLS = "_urls"


def contar_tokens(texto: str, modelo: Optional[str] = None) -> int:
    """Cuenta tokens con el tokenizador del modelo de embeddings."""
    modelo = modelo or obtener_ajustes().modelo_embeddings
    try:
        codificador = tiktoken.encoding_for_model(modelo)
    except KeyError:
        codificador = tiktoken.get_encoding("cl100k_base")
    return len(codificador.encode(texto))


def _quitar_hipervinculos(texto: str) -> tuple[str, dict[str, str]]:
    """Deja las citas en texto plano y devuelve el mapa cita -> URL.

    Cada URL de la CSJN pesa ~120 caracteres. Dejarlas en el texto que se indexa
    ensucia el embedding y, en los extractos muy citados, hace que la sola linea de
    fuentes supere el techo de tokens: ahi el splitter termina separando la doctrina
    de su cita, que es justo lo que este diseño quiere evitar.
    """
    urls: dict[str, str] = {}

    def reemplazar(coincidencia: re.Match) -> str:
        cita, url = coincidencia.group(1), coincidencia.group(2)
        urls.setdefault(cita, url)
        return cita

    return PATRON_LINK_MD.sub(reemplazar, texto), urls


def leer_documentos(directorio: Path = cfg.DIRECTORIO_DATA) -> list[Document]:
    """Un Document por archivo del dataset, con su seccion y sus URLs como metadato."""
    archivos = sorted(directorio.glob(cfg.PATRON_DOCUMENTOS))
    if not archivos:
        # `ErrorRAG` y no `FileNotFoundError`: el worker atrapa los errores del proyecto, y
        # una excepción sin traducir lo mata con un stack trace en vez del mensaje preparado.
        raise ErrorRAG(
            cfg.MENSAJE_SIN_DATASET.format(patron=cfg.PATRON_DOCUMENTOS, ruta=directorio)
        )
    documentos = []
    for archivo in archivos:
        try:
            contenido = archivo.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise ErrorRAG(
                cfg.ERROR_LECTURA_DATASET.format(archivo=archivo.name, detalle=error)
            ) from error
        if not contenido.strip():
            raise ErrorRAG(cfg.ERROR_DOCUMENTO_VACIO.format(archivo=archivo.name))
        texto, urls = _quitar_hipervinculos(contenido)
        titulo = texto.splitlines()[0].lstrip("# ").strip()
        documentos.append(
            Document(
                page_content=texto,
                metadata={"origen": archivo.name, "seccion": titulo, CLAVE_URLS: urls},
            )
        )
    return documentos


def crear_splitter() -> RecursiveCharacterTextSplitter:
    """Splitter que mide en tokens y corta primero en el limite entre extractos.

    `add_start_index` hace que cada fragmento sepa en que posicion del documento
    empieza: buscarlo despues por su texto no sirve, porque la doctrina repite
    parrafos casi identicos y la busqueda cae en la subseccion equivocada.
    """
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        separators=cfg.SEPARADORES,
        chunk_size=cfg.TAMANO_CHUNK_TOKENS,
        chunk_overlap=cfg.SOLAPAMIENTO_CHUNK_TOKENS,
        keep_separator=True,
        add_start_index=True,
    )


def _subtitulo_vigente(texto_completo: str, posicion: int) -> str:
    """Ultimo subtitulo Markdown que precede a esa posicion del documento.

    Si ningun subtitulo precede a la posicion (el primer fragmento arranca antes
    del primer `###`), vale el primer subtitulo del documento: en este dataset el
    contenido bajo el titulo principal pertenece siempre a esa primera
    subseccion, y `subseccion` es clave de busqueda de fallos_citados — un valor
    vacio dejaria fragmentos (y sus citas) inalcanzables.
    """
    titulos = [(m.start(), m.group(1)) for m in PATRON_SUBTITULO.finditer(texto_completo)]
    vigente = ""
    for inicio, titulo in titulos:
        if inicio <= posicion:
            vigente = titulo
        else:
            break
    if not vigente and titulos:
        vigente = titulos[0][1]
    return vigente


def _posicion_en_documento(completo: str, fragmento: Document) -> int:
    """Donde empieza el fragmento dentro de su documento.

    `start_index` es la fuente principal, pero el splitter devuelve -1 cuando el
    fragmento no aparece literal en el original (pasa con los que arrancan en el
    separador de extractos): ahi se busca por el texto sin el separador.
    """
    posicion = fragmento.metadata.get("start_index", -1)
    if posicion >= 0:
        return posicion
    aguja = fragmento.page_content.lstrip("-# \n")[:120]
    return max(completo.find(aguja), 0)


def fragmentar(documentos: list[Document]) -> list[Document]:
    """Aplica el splitter y enriquece cada fragmento con su subtitulo y sus citas."""
    splitter = crear_splitter()
    fragmentos = splitter.split_documents(documentos)
    texto_por_origen = {doc.metadata["origen"]: doc.page_content for doc in documentos}

    for fragmento in fragmentos:
        completo = texto_por_origen[fragmento.metadata["origen"]]
        posicion = _posicion_en_documento(completo, fragmento)
        # Chroma solo acepta metadatos escalares: el mapa de URLs de las citas que
        # aparecen en ESTE fragmento viaja serializado.
        urls_documento = fragmento.metadata.pop(CLAVE_URLS, {})
        citas = {
            cita.strip()
            for linea in PATRON_CITAS.findall(fragmento.page_content)
            for cita in linea.split(";")
            if cita.strip()
        }
        fragmento.metadata["citas_urls"] = json.dumps(
            {cita: urls_documento[cita] for cita in citas if cita in urls_documento},
            ensure_ascii=False,
        )
        fragmento.metadata["start_index"] = posicion
        fragmento.metadata["subseccion"] = _subtitulo_vigente(completo, posicion)
        fragmento.metadata["tokens"] = contar_tokens(fragmento.page_content)
    return fragmentos


def indice_ya_poblado(vectorstore: Chroma) -> int:
    """Cantidad de fragmentos persistidos. Levanta ErrorRAG si la base falla.

    `_collection.count()` es privado de langchain-chroma y cuenta sin traer nada; un upgrade
    podría sacarlo. El `AttributeError` cae al camino público, que pide los ids y los cuenta:
    más caro, pero sobrevive a que la librería cambie por dentro.
    """
    try:
        try:
            return vectorstore._collection.count()
        except AttributeError:
            return len(vectorstore.get(include=[])["ids"])
    except (ChromaError, OSError, KeyError) as error:
        raise ErrorRAG(cfg.ERROR_VECTORSTORE.format(detalle=error)) from error


def abrir_vectorstore() -> Chroma:
    """Abre (o crea) la coleccion persistente con el modelo de embeddings del proyecto.

    Construir el cliente no hace I/O de red —el embedding recien se calcula al indexar
    o al consultar—, asi que lo que puede fallar aca es la base en disco: sqlite
    bloqueado, ruta invalida, o una coleccion previa con otra dimension de vectores.
    """
    ajustes = obtener_ajustes()
    try:
        return Chroma(
            collection_name=ajustes.nombre_coleccion,
            embedding_function=crear_embeddings(ajustes),
            persist_directory=str(ajustes.directorio_vectorstore),
            collection_metadata=cfg.METRICA_DISTANCIA,
        )
    except (OpenAIError, ErrorDeConfiguracion) as error:
        raise ErrorRAG(str(error)) from error
    except (ChromaError, OSError, ValueError) as error:
        raise ErrorRAG(cfg.ERROR_VECTORSTORE.format(detalle=error)) from error


def poblar_indice(vectorstore: Chroma, fragmentos: list[Document]) -> int:
    """Indexa los fragmentos y devuelve cuantos quedaron en la coleccion."""
    try:
        vectorstore.add_documents(fragmentos)
    except RateLimitError as error:
        raise ErrorRAG(cfg.ERROR_LIMITE.format(detalle=error)) from error
    except APIConnectionError as error:
        raise ErrorRAG(cfg.ERROR_CONEXION.format(detalle=error)) from error
    except APIError as error:
        raise ErrorRAG(cfg.ERROR_API.format(detalle=error)) from error
    except (ChromaError, OSError) as error:
        raise ErrorRAG(cfg.ERROR_VECTORSTORE.format(detalle=error)) from error
    return indice_ya_poblado(vectorstore)


def asegurar_indice(verboso: bool = True) -> Chroma:
    """Devuelve la coleccion lista para consultar, indexando solo si hace falta.

    El chequeo anti-reindexado es el que evita pagar embeddings de nuevo en cada corrida.
    """
    cargar_entorno()
    vectorstore = abrir_vectorstore()
    existentes = indice_ya_poblado(vectorstore)

    if existentes:
        if verboso:
            print(
                cfg.MENSAJE_INDICE_EXISTENTE.format(
                    ruta=obtener_ajustes().directorio_vectorstore.name, cantidad=existentes
                )
            )
        return vectorstore

    if verboso:
        print(cfg.MENSAJE_INDEXANDO.format(ruta=obtener_ajustes().directorio_vectorstore.name))
    documentos = leer_documentos()
    if verboso:
        tokens_totales = sum(contar_tokens(doc.page_content) for doc in documentos)
        print(cfg.MENSAJE_DOCUMENTOS_LEIDOS.format(cantidad=len(documentos), tokens=tokens_totales))

    fragmentos = fragmentar(documentos)
    if verboso:
        medidas = [fragmento.metadata["tokens"] for fragmento in fragmentos]
        print(
            cfg.MENSAJE_CHUNKS.format(
                cantidad=len(fragmentos),
                minimo=min(medidas),
                promedio=round(sum(medidas) / len(medidas)),
                maximo=max(medidas),
                techo=cfg.TAMANO_CHUNK_TOKENS,
            )
        )

    total = poblar_indice(vectorstore, fragmentos)
    if verboso:
        print(cfg.MENSAJE_INDICE_LISTO.format(cantidad=total, coleccion=obtener_ajustes().nombre_coleccion))
    return vectorstore


if __name__ == "__main__":
    try:
        asegurar_indice()
    except ErrorRAG as error:
        print(error)
        raise SystemExit(1)

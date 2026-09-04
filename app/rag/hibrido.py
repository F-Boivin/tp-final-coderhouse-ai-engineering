"""Recuperación híbrida: BM25 léxico y búsqueda vectorial, fusionados por rango.

Los dos retrievers ven el mismo corpus —los fragmentos que ya están en el índice— así que
describen la misma base y la fusión dedupica por contenido. El léxico encuentra la cita
escrita literal, que el embedding diluye entre párrafos parecidos; el vectorial encuentra la
consulta parafraseada, que el léxico no matchea. `crear_hibrido` los pesa por igual.

El lado vectorial es async y emite sus dos spans: sin ellos, la recuperación sería un hueco
de tiempo adentro de la herramienta.

Verificado contra la documentación oficial:
- `EnsembleRetriever` vive en `langchain_classic` desde la v1 y fusiona por rango recíproco.
  https://reference.langchain.com/python/langchain-classic/retrievers/ensemble/EnsembleRetriever
- `BM25Retriever.from_documents(..., preprocess_func=...)` aplica ese tokenizador al corpus
  y a la consulta.
  https://reference.langchain.com/python/langchain-community/retrievers/bm25/BM25Retriever
"""

from typing import Optional

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

import app.nucleo.constantes as cfg
from app.nucleo.config import obtener_ajustes
from app.observabilidad.trazas import (
    anotar_documentos,
    span_de_embedding,
    span_de_recuperacion,
)
from app.rag.ingesta import contar_tokens


def tokenizar(texto: str) -> list[str]:
    """Separa en palabras y despega la puntuación que se pega a las citas.

    Sin esto, `316:2343;` y `316:2343` son términos distintos para BM25 y la cita escrita al
    final de una enumeración deja de matchear.
    """
    return [token.strip(";,.()") for token in texto.split()]


class RecuperadorVectorial(BaseRetriever):
    """El lado vectorial del ensamble, con la búsqueda partida en sus dos mitades medibles.

    Embeber la consulta es una llamada de red al proveedor; buscar el vecino más cercano es
    trabajo local del índice. Medirlas juntas da un número que no se puede atribuir.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: Chroma
    k: int = cfg.RESULTADOS_RECUPERADOS

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Los `k` fragmentos más cercanos a la consulta."""
        if self.vectorstore.embeddings is None:
            raise RuntimeError(cfg.ERROR_HERRAMIENTA_SIN_EMBEDDINGS)
        tokens = contar_tokens(query)
        with span_de_embedding("embeber_consulta", query,
                               obtener_ajustes().modelo_embeddings, tokens):
            vector = await self.vectorstore.embeddings.aembed_query(query)
        with span_de_recuperacion("chroma_vecinos", query, k=self.k) as span:
            documentos = await self.vectorstore.asimilarity_search_by_vector(vector, k=self.k)
            anotar_documentos(span, documentos)
        return documentos

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """El camino sincrónico, que este retriever no ofrece."""
        # `BaseRetriever` lo declara abstracto, así que existe para fallar ruidoso: nadie del
        # sistema lo llama, y quien lo llamara bloquearía el event loop en silencio.
        raise NotImplementedError(cfg.ERROR_RECUPERADOR_SINCRONICO)


def crear_hibrido(vectorstore: Chroma, documentos: list[Document],
                  top_k: Optional[int] = None) -> EnsembleRetriever:
    """El ensamble de los dos retrievers sobre el mismo corpus.

    Cada lado aporta más candidatos que los que se van a devolver: la fusión elige mejor
    viendo más de cada uno, y el recorte a `top_k` lo aplica quien consulta, después de
    fusionar.
    """
    top_k = top_k or cfg.RESULTADOS_RECUPERADOS
    candidatos = max(cfg.CANDIDATOS_POR_RETRIEVER, top_k)
    bm25 = BM25Retriever.from_documents(documentos, k=candidatos, preprocess_func=tokenizar)
    vectorial = RecuperadorVectorial(vectorstore=vectorstore, k=candidatos)
    return EnsembleRetriever(
        retrievers=[bm25, vectorial],
        weights=[cfg.PESO_BM25, cfg.PESO_VECTORIAL],
    )

"""El cableado del grafo: quién puede seguir a quién, y con qué condición.

El supervisor decide y los tres especialistas devuelven el control. Las dos aristas
condicionales traducen esa decisión en un destino: `enrutar` manda al especialista elegido,
al nodo de aprobación o al final; `enrutar_aprobacion` decide si un trabajo publicado cierra
o vuelve al supervisor.

Verificado contra la documentación oficial (langgraph 1.2.9):
- Sin el mapeo explícito de `add_conditional_edges`, un destino inexistente se loguea y el
  grafo termina como si nada, sin levantar excepción.
  https://reference.langchain.com/python/langgraph/graph/state/StateGraph/add_conditional_edges
- `recursion_limit` va en el config de la invocación y no en `compile()`; cuenta supersteps.
  https://docs.langchain.com/oss/python/langgraph/graph-api#recursion-limit
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph

import app.nucleo.constantes as cfg
from app.agentes.analyst_agent import verificador_node
from app.agentes.research_agent import investigador_node
from app.agentes.writer_agent import redactor_node
from app.grafo.estado import (
    APROBACION,
    DESTINOS,
    INVESTIGADOR,
    NODOS,
    REDACTOR,
    SUPERVISOR,
    VERIFICADOR,
    EstadoOrquestador,
    leer_situacion,
)
from app.grafo.hitl import aprobacion_node
from app.grafo.supervisor import supervisor_node
from app.nucleo.errores import ErrorDeAgente


def falta_publicar(state: EstadoOrquestador) -> bool:
    """Si hay una respuesta terminada que todavía no se publicó.

    Es la única condición para entrar al nodo de publicación. Si ese trabajo además necesita
    que lo mire una persona lo decide el nodo, midiendo su calidad.

    Cuando el supervisor cierra por tope de vueltas o de intentos, `listo` es falso y el grafo
    termina sin pasar por acá: un trabajo que agotó los frenos no puede quedar esperando a
    una persona.
    """
    return leer_situacion(state).listo and not state.get("publicado")


def enrutar(
    state: EstadoOrquestador,
) -> Literal["investigador", "verificador", "redactor", "aprobacion", "__end__"]:
    """Traduce la decisión del supervisor a un destino del grafo.

    En el Literal va el VALOR de END (`"__end__"`) porque el tipo describe lo que la función
    devuelve, y es lo que después dibuja bien el diagrama.

    El `.strip()` normaliza antes de decidir. Sin él, un destino con un espacio de más no
    coincide con ninguna clave del mapeo y LangGraph revienta con un KeyError crudo que no
    menciona el ruteo.
    """
    # `completado` es el veredicto explicito que escribe el supervisor: la arista lo lee y
    # no vuelve a deducir si la tarea termino. `siguiente` queda como el destino cuando no
    # termino, y los dos se escriben juntos en el mismo return del nodo.
    if state.get("completado") or str(state.get("siguiente", "")).strip() == "FINALIZAR":
        # El cierre pasa por el nodo de publicación, no por el supervisor: un LLM que pudiera
        # elegir saltearlo haría que la pausa dejara de ser obligatoria.
        return APROBACION if falta_publicar(state) else END
    destino = str(state.get("siguiente", "")).strip()
    if destino not in DESTINOS:
        raise ErrorDeAgente(cfg.ERROR_DESTINO_INVALIDO.format(destino=destino))
    return destino


def enrutar_aprobacion(state: EstadoOrquestador) -> Literal["supervisor", "__end__"]:
    """Traduce en un destino lo que resolvió el nodo de publicación.

    Traduce una decisión ya tomada, sin elegir nada. Publicado cierra —lo haya decidido una
    persona o el propio criterio de calidad—. Las otras dos acciones vuelven al supervisor,
    que lee `ampliacion_pendiente` o `reescritura_pendiente` y manda al investigador o al
    redactor según cuál sea.
    """
    return END if state.get("publicado") else SUPERVISOR


def crear_grafo(
    investigador=investigador_node,
    verificador=verificador_node,
    redactor=redactor_node,
    *,
    supervisor=supervisor_node,
    aprobacion=aprobacion_node,
    checkpointer=None,
):
    """Arma el orquestador y lo compila.

    Los cinco nodos entran por parámetro con su implementación real por defecto. No es un
    gancho de framework: es lo que permite que la demostración del ciclo de corrección
    reemplace un nodo sin volver a escribir el cableado. Cuando la demo armaba su propio
    grafo, las aristas quedaban declaradas dos veces y nada garantizaba que siguieran siendo
    las mismas. Un script que corra sin humano pasa un `aprobacion=` que apruebe solo.

    Sin `checkpointer` el grafo no puede pausarse: `interrupt()` lo exige.
    """
    grafo = StateGraph(EstadoOrquestador)

    grafo.add_node(SUPERVISOR, supervisor)
    grafo.add_node(INVESTIGADOR, investigador)
    grafo.add_node(VERIFICADOR, verificador)
    grafo.add_node(REDACTOR, redactor)
    grafo.add_node(APROBACION, aprobacion)

    grafo.add_edge(START, SUPERVISOR)
    # Aristas fijas: los especialistas no deciden nada, siempre devuelven el control.
    for nodo in NODOS:
        grafo.add_edge(nodo, SUPERVISOR)

    # El mapeo va explícito aunque la API lo acepte omitido: sin él, un destino que no
    # existe se ignora en silencio y el grafo termina como si hubiera funcionado.
    grafo.add_conditional_edges(
        SUPERVISOR,
        enrutar,
        {**{nodo: nodo for nodo in NODOS}, APROBACION: APROBACION, END: END},
    )
    # La segunda condicional. El supervisor sigue siendo el único que elige entre
    # especialistas: esta no elige, traduce un veredicto humano ya tomado.
    grafo.add_conditional_edges(
        APROBACION,
        enrutar_aprobacion,
        {SUPERVISOR: SUPERVISOR, END: END},
    )
    return grafo.compile(checkpointer=checkpointer)

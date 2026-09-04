"""Agente de investigación: arma una síntesis con sus citas encadenando dos herramientas.

`buscar_doctrina` trae la doctrina y lista los nombres de subsección; `fallos_citados` devuelve,
para cada una, la lista autorizada de fallos. El prompt pide ese orden y nada en los datos lo
fuerza: los fragmentos ya traen números de fallo, así que el agente podría citar sin la segunda
llamada. La garantía la sostiene el verificador, no el orden.

Devuelve un `Investigacion` de Pydantic, con la síntesis y las citas en campos separados para
que el verificador compruebe una por una.
"""

from langgraph.prebuilt import create_react_agent

import app.nucleo.constantes as cfg
import app.rag.herramientas as herramientas
from app.nucleo.modelos import crear_chat
from app.nucleo.errores import ErrorDeAgente
from app.grafo.estado import (
    EstadoOrquestador,
    Investigacion,
    leer_situacion,
    ultima_aprobacion,
    ultima_verificacion,
)

PROMPT = """Sos el especialista en investigación jurídica del equipo. Tu única fuente de \
verdad es el cuadernillo de doctrina de la CSJN sobre sentencias arbitrarias, indexado en \
tus herramientas.

Cómo trabajás:
1. Buscá la doctrina con `buscar_doctrina`. Si la consulta abarca más de un tema, hacé una \
búsqueda por tema.
2. Traé las citas con `fallos_citados`, usando los nombres de subsección que \
`buscar_doctrina` lista al final. Los fallos que menciones tienen que salir de ahí.
3. Escribí una síntesis que responda la consulta, y separá cada afirmación con su fallo.

Reglas que no se negocian:
- Nunca inventes un número de fallo. Si no lo trajo una herramienta, no existe.
- Cada cita tiene que decir de qué subsección salió.
- Si te devuelven una investigación rechazada, corregí exactamente lo que se te señala: \
sacá las citas inexistentes y buscá respaldo real para esas afirmaciones, o quitá la \
afirmación que no podés sostener."""


def construir_investigador():
    """Arma el agente ReAct de investigación.

    `response_format` obliga a que la salida final sea un `Investigacion` válido: si el
    modelo devuelve algo que no cumple el esquema, falla acá y no tres nodos más adelante,
    cuando el dato ya se propagó.
    """
    modelo = crear_chat("investigador")
    return create_react_agent(
        model=modelo,
        tools=herramientas.HERRAMIENTAS_INVESTIGACION,
        prompt=PROMPT,
        response_format=Investigacion,
    )


async def investigador_node(state: EstadoOrquestador) -> dict:
    """Investiga y agrega su artefacto al estado.

    Recibe el contexto mínimo: la consulta y, si lo hubo, el motivo del rechazo anterior. La
    consigna advierte sobre la contaminación de contexto, y saber cómo el supervisor tomó sus
    decisiones no le sirve para investigar mejor.
    """
    situacion = leer_situacion(state)
    pedido = f"Consulta a investigar: {state['consulta']}"
    rechazo = ultima_verificacion(state)
    if rechazo is not None and not rechazo.aprobado:
        pedido += (
            "\n\nTu investigación anterior fue rechazada por el verificador de citas.\n"
            f"Fallos que NO existen en el corpus: {', '.join(rechazo.inexistentes) or '(ninguno)'}\n"
            "Observaciones:\n- " + "\n- ".join(rechazo.observaciones)
        )

    # Un revisor humano puede pedir más material aunque el verificador haya aprobado todo. Es
    # otro pedido que el anterior: ahí las citas estaban mal, acá son pocas.
    #
    # La vigencia sale de `leer_situacion` y no de la aprobación cruda: un pedido de ampliar
    # ya cumplido sigue siendo la última aprobación del estado, y leerlo directo le sumaría
    # "buscá más doctrina" a un investigador que solo tiene que corregir citas. Es la misma
    # lectura que usan el supervisor y los frenos, así que las tres dicen lo mismo.
    humana = ultima_aprobacion(state)
    if situacion.ampliacion_pendiente and humana is not None:
        pedido += (
            "\n\nUn revisor humano leyó la respuesta y pidió MÁS DOCTRINA antes de publicarla.\n"
            "Buscá en otras subsecciones del corpus y sumá fallos que no hayas citado todavía.\n"
            "Motivos:\n- " + "\n- ".join(humana.motivos)
        )

    try:
        salida = await construir_investigador().ainvoke(
            {"messages": [("user", pedido)]},
            {"recursion_limit": cfg.LIMITE_RECURSION_AGENTE},
        )
    except Exception as exc:  # el SDK y LangGraph traen sus propias jerarquías
        raise ErrorDeAgente(cfg.ERROR_INVESTIGADOR.format(detalle=f"{type(exc).__name__}: {exc}")) from exc

    investigacion = salida.get("structured_response")
    if investigacion is None:
        raise ErrorDeAgente(cfg.ERROR_INVESTIGADOR.format(detalle="no devolvió un artefacto estructurado"))

    # El conteo se arma acá y no viaja en el artefacto: `Investigacion` lo produce el modelo
    # vía `response_format`, así que un campo más sería un campo que el modelo tendría que
    # llenar. Nadie más rearma este mensaje, así que no hay dos lugares que puedan discrepar.
    llamadas = herramientas.contar_llamadas(salida.get("messages") or [])
    return {
        "investigaciones": (investigacion,),
        "messages": [
            (
                "assistant",
                f"[investigador] {llamadas} llamadas a herramientas · "
                f"{len(investigacion.citas)} citas sobre "
                f"{len(investigacion.subsecciones)} subsecciones.",
            )
        ],
    }

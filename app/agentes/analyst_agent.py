"""Nodo de análisis: dictamina si cada cita existe comparándola contra el padrón del corpus.

Es código y no llama a ningún LLM: un fallo está entre las 574 del padrón o no está, y eso se
computa. Poner un modelo a juzgarlo mudaría la alucinación al que audita. Por eso es un nodo y
no un agente — la documentación reserva ese nombre para "a model calling tools in a loop":
https://docs.langchain.com/oss/python/langchain/agents

El archivo conserva el nombre `analyst_agent.py` que propone la consigna. El README explica por
qué es un nodo del grafo y no una herramienta del supervisor.
"""

import app.nucleo.constantes as cfg
import app.rag.herramientas as herramientas
from app.nucleo.errores import ErrorDeAgente
from app.grafo.estado import EstadoOrquestador, Investigacion, Verificacion, ultima_investigacion


async def verificador_node(state: EstadoOrquestador) -> dict:
    """Contrasta las citas de la última investigación contra el padrón del corpus."""
    investigacion = ultima_investigacion(state)
    if investigacion is None:
        raise ErrorDeAgente(cfg.ERROR_VERIFICADOR_SIN_INVESTIGACION)

    veredicto = await verificar(investigacion)
    return {
        "verificaciones": (veredicto,),
        "messages": [
            (
                "assistant",
                f"[verificador] {len(veredicto.verificadas)} de "
                f"{len(veredicto.verificadas) + len(veredicto.inexistentes)} citas existen en el corpus. "
                f"{'APROBADO' if veredicto.aprobado else 'RECHAZADO'}.",
            )
        ],
    }


async def verificar(investigacion: Investigacion) -> Verificacion:
    """Devuelve el veredicto sobre una investigación.

    Separada del nodo para poder probarla sola, sin montar el grafo.
    """
    if not investigacion.citas:
        # Una síntesis sin citas es inverificable, que es distinto de tener las citas mal:
        # se rechaza diciendo exactamente eso.
        return Verificacion(
            aprobado=False,
            observaciones=("la sintesis no trae ninguna cita: no hay nada que verificar",),
        )

    afirmadas = [c.fallo for c in investigacion.citas]

    # Una cita por afirmacion. Si el `fallo` amontona varias -"Fallos: 313:1045; 328:4597",
    # que es como las escribe el cuadernillo-, normalizarlo se queda con la primera y las
    # demas quedarian por comprobadas sin que nadie las haya mirado.
    amontonadas = sorted({c.fallo for c in investigacion.citas
                          if herramientas.cuantas_citas(c.fallo) > 1})

    # Un campo `fallo` sin ningún número reconocible es otro problema, y necesita otro
    # mensaje: decirle "separá las citas" a quien no escribió ninguna lo manda a corregir algo
    # que no hizo.
    sin_numero = sorted({c.fallo for c in investigacion.citas
                         if herramientas.cuantas_citas(c.fallo) == 0})

    # Las citas escritas en la prosa de la sintesis se comprueban igual que las de la lista.
    # Un verificador que solo mira la lista deja pasar cualquier numero escrito al costado, y
    # esa sintesis es despues el material del que redacta el ultimo agente.
    de_la_sintesis = sorted(herramientas.citas_del_texto(investigacion.sintesis))

    a_comprobar = list(dict.fromkeys(afirmadas + de_la_sintesis))
    try:
        crudo = await herramientas.verificar_citas.ainvoke({"citas": a_comprobar})
    except Exception as exc:
        raise ErrorDeAgente(cfg.ERROR_VERIFICADOR.format(detalle=f"{type(exc).__name__}: {exc}")) from exc

    veredicto = herramientas.leer_veredicto(crudo)
    faltantes = [c for c in a_comprobar if c not in veredicto]
    if faltantes:
        # Un veredicto que no cubre todas las citas es una respuesta rota, y leerlo como
        # "no existen" seria inventar el peor veredicto posible.
        raise ErrorDeAgente(
            cfg.ERROR_VERIFICADOR.format(
                detalle=f"la respuesta no cubre {len(faltantes)} de {len(a_comprobar)} citas"
            )
        )
    verificadas = tuple(c for c in afirmadas if veredicto.get(c))
    inexistentes = tuple(c for c in afirmadas if not veredicto.get(c))
    sueltas_inventadas = tuple(c for c in de_la_sintesis if not veredicto.get(c))

    observaciones = [
        f"'{cita.fallo}' no figura entre las citas del corpus; sostiene: {cita.afirmacion[:90]}"
        for cita in investigacion.citas
        if cita.fallo in inexistentes
    ]
    for cita in amontonadas:
        observaciones.append(
            f"'{cita}' junta varias citas en una: escribi una cita por afirmacion, con un "
            f"solo numero de fallo"
        )
    for cita in sin_numero:
        observaciones.append(
            f"'{cita}' no tiene ningun numero de fallo: escribilo en la forma "
            f"\"Fallos: tomo:pagina\""
        )
    for cita in sueltas_inventadas:
        observaciones.append(
            f"la sintesis menciona el fallo '{cita}', que no figura en el corpus: sacalo del "
            f"texto o reemplazalo por uno real"
        )

    # Atribuir una cita cierta a una subseccion que no existe tambien es fabricar: la
    # cita queda sin procedencia comprobable, que es justo lo que el sistema promete.
    try:
        inventadas = sorted({
            c.subseccion for c in investigacion.citas
            if c.subseccion and not herramientas.subseccion_existe(c.subseccion)
        })
    except Exception as exc:
        # Sin este except una caida de la base aca sale cruda y saltea la traduccion que el
        # resto de la funcion se toma el trabajo de hacer.
        raise ErrorDeAgente(cfg.ERROR_VERIFICADOR.format(detalle=f"{type(exc).__name__}: {exc}")) from exc
    for nombre in inventadas:
        observaciones.append(
            f"la subseccion '{nombre}' no existe en el corpus: usa el nombre exacto que "
            f"devuelve buscar_doctrina, con su numeracion"
        )

    aprobado = not (inexistentes or inventadas or amontonadas or sin_numero
                    or sueltas_inventadas)
    if aprobado and len(verificadas) < cfg.CITAS_MINIMAS:
        # Todas ciertas pero pocas: la respuesta es correcta y floja. Es un rechazo
        # distinto del anterior, y el investigador tiene que poder distinguirlos.
        aprobado = False
        observaciones.append(
            f"solo {len(verificadas)} cita(s) verificada(s); se piden al menos "
            f"{cfg.CITAS_MINIMAS} para dar la respuesta por fundada"
        )

    return Verificacion(
        verificadas=verificadas,
        inexistentes=inexistentes,
        aprobado=aprobado,
        observaciones=tuple(observaciones),
    )

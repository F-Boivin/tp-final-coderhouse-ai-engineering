"""El worker: el único proceso que corre el grafo.

Toma trabajos de la cola de Redis, los ejecuta, y los deja en su estado final. Cuando el grafo
pausa para pedir aprobación humana, guarda la pausa y suelta el trabajo: reanudarlo es otro
turno de cola, no una espera con el consumidor bloqueado.

Adentro corre `CONSUMIDORES` corrutinas sobre un mismo event loop. Todos los nodos son `async`,
así que la concurrencia es real con una sola apertura de Chroma, un padrón, un cliente de
Redis y un checkpointer. Escalar a varios contenedores sigue disponible —`--scale worker=3`—
justamente porque la cola vive en Redis.

Uso:
    python -m app.trabajos.worker
"""

import asyncio
import signal
import sys

from langgraph.errors import GraphRecursionError
from langgraph.types import Command

import app.nucleo.constantes as cfg
import app.rag.herramientas as herramientas
import app.observabilidad.langsmith as observability
import app.trabajos.tareas as tareas
from app.nucleo.errores import (
    ErrorDeAgente,
    ErrorDeAlmacenamiento,
    ErrorDeConfiguracion,
    ErrorRAG,
)
from app.grafo.construccion import crear_grafo
from app.nucleo.config import cargar_entorno, obtener_ajustes
from app.nucleo.modelos import modelo_del_rol
from app.rag.ingesta import asegurar_indice
from app.trabajos.tareas import PedidoDeAprobacion, Tarea


def registrar(mensaje: str) -> None:
    """Escribe una línea de log con marca de tiempo.

    Con `flush`: la salida de un contenedor está redirigida a un pipe, y sin vaciarla los
    mensajes aparecen a los miles de caracteres en vez de cuando pasan las cosas.
    """
    print(f"{tareas.ahora()} · {mensaje}", flush=True)


def estado_inicial(consulta: str) -> dict:
    """El estado con el que arranca un trabajo nuevo."""
    return {
        "messages": [],
        "consulta": consulta,
        "siguiente": "investigador",
        "investigaciones": (),
        "verificaciones": (),
        "redacciones": (),
        "aprobaciones": (),
        "intentos": 0,
        "vueltas": 0,
        "completado": False,
        "publicado": False,
    }


def config_de(tarea: Tarea) -> dict:
    """El config con el que se invoca el grafo para este trabajo.

    El `thread_id` es lo que hace que el checkpoint del tramo anterior se recupere al reanudar.
    El `metadata` es lo que después permite juntar en el dashboard las dos trazas de un mismo
    trabajo: la pausa las parte en dos ejecuciones, y sin el `job_id` en ambas quedan como dos
    corridas sin relación. El proveedor y los modelos van ahí para que la traza diga con
    qué se ejecutó, sin abrir un nodo.
    """
    ajustes = obtener_ajustes()
    return {
        "configurable": {"thread_id": tarea.thread_id},
        "recursion_limit": cfg.LIMITE_RECURSION_ORQUESTADOR,
        "run_name": f"orquestador·{tarea.job_id[:8]}",
        "metadata": {
            "job_id": tarea.job_id,
            "revision": tarea.revisiones,
            "proveedor": ajustes.llm_provider.value,
            "modelos": {rol: modelo_del_rol(rol, ajustes)
                        for rol in ("supervisor", "investigador", "redactor")},
        },
    }


async def entrada_del_grafo(cliente, saver, tarea: Tarea):
    """Qué se le pasa al grafo, según cómo llegó el trabajo a la cola.

    Tres caminos. Con un veredicto humano pendiente, el `Command` que lo reanuda. Sin
    veredicto pero con un checkpoint previo —un trabajo que otro worker dejó a medias—,
    `None`: LangGraph sigue desde donde quedó, mientras que reinyectar el estado inicial
    pisaría `vueltas` e `intentos`, que no tienen reducer. Y un trabajo nuevo arranca de cero.

    Un trabajo en `awaiting_approval` vuelve a la cola recién cuando la API escribió la
    decisión, así que acá ya tiene que estar. Si no está, el trabajo se reencoló por otro
    motivo y arrancar de cero perdería lo hecho: es un error, no un caso a tolerar.
    """
    if tarea.aprobacion is not None:
        veredicto = await tareas.leer_decision(
            cliente, tarea.job_id, tarea.aprobacion.interrupt_id
        )
        if veredicto is None:
            raise ErrorDeAlmacenamiento(
                cfg.ERROR_DECISION_AUSENTE.format(job_id=tarea.job_id)
            )
        return Command(resume=veredicto)

    tupla = await saver.aget_tuple({"configurable": {"thread_id": tarea.thread_id}})
    if tupla is not None:
        return None
    return estado_inicial(tarea.consulta)


async def correr_tramo(grafo, entrada, config) -> None:
    """Corre el grafo hasta que termine o pause, mostrando lo que cada nodo decide.

    El `continue` sobre `__interrupt__` no es cosmético: la actualización de una pausa es una
    tupla de `Interrupt`, y el `.get("messages")` de abajo la trataría como diccionario y
    reventaría con un AttributeError que no menciona al HITL.
    """
    async for modo, dato in grafo.astream(
        entrada, config, stream_mode=["updates", "values"]
    ):
        if modo != "updates":
            continue
        for nodo, actualizacion in dato.items():
            if nodo == "__interrupt__" or not isinstance(actualizacion, dict):
                continue
            for mensaje in actualizacion.get("messages", []):
                texto = mensaje[1] if isinstance(mensaje, tuple) else mensaje.content
                registrar(f"    {texto}")


async def cerrar(cliente, grafo, tarea: Tarea, config) -> Tarea:
    """Lee el estado final del grafo y lo traduce al estado del trabajo.

    La fuente es `aget_state` y no lo que devolvió el stream: viene del checkpoint en Redis, así
    que **cualquier** worker puede leerlo, no solo el que empezó el trabajo.

    Un trabajo que agotó los frenos sin publicar termina en `completed` con `publicado` en
    falso. Es un desenlace del sistema funcionando, y por eso no es `failed`.
    """
    snapshot = await grafo.aget_state(config)
    estado = snapshot.values

    if snapshot.interrupts:
        pausa = snapshot.interrupts[0]
        pedido = PedidoDeAprobacion(
            interrupt_id=pausa.id,
            # El reloj lo pone el worker y no el nodo: el nodo se re-ejecuta al reanudar y
            # daría dos valores distintos para el mismo pedido.
            solicitada_en=tareas.ahora(),
            **pausa.value,
        )
        # Un worker que muere entre la decisión de la API y el final de la reanudación deja
        # la pausa abierta con su veredicto ya escrito. Guardarlo como `awaiting_approval` lo
        # dejaría esperando para siempre: el `SET NX` está tomado y nadie podría volver a
        # decidir. Se re-encola con la decisión que ya existe.
        if await tareas.leer_decision(cliente, tarea.job_id, pausa.id) is not None:
            registrar(f"[{tarea.job_id}] la pausa {pausa.id} ya tiene decisión: vuelve a la cola")
            tarea = await tareas.guardar(
                cliente, tarea.con(estado=cfg.PENDIENTE, aprobacion=pedido))
            await tareas.encolar(cliente, tarea.job_id)
            return tarea

        registrar(f"[{tarea.job_id}] pausa para aprobación humana ({pausa.id})")
        return await tareas.guardar(cliente, tarea.con(
            estado=cfg.ESPERANDO_APROBACION,
            revisiones=len(estado.get("aprobaciones") or ()),
            aprobacion=pedido,
        ))

    redacciones = estado.get("redacciones") or ()
    final = redacciones[-1] if redacciones else None
    publicado = bool(estado.get("publicado"))
    registrar(f"[{tarea.job_id}] cerrado · publicado={publicado}")
    return await tareas.guardar(cliente, tarea.con(
        estado=cfg.COMPLETADA,
        resultado=final.texto if final else None,
        citas=final.citas_usadas if final else (),
        publicado=publicado,
        revisiones=len(estado.get("aprobaciones") or ()),
        aprobacion=None,
    ))


async def fallar(cliente, tarea: Tarea, detalle: str) -> Tarea:
    """Deja el trabajo en `failed` con el motivo, que es lo que después devuelve el endpoint."""
    registrar(f"[{tarea.job_id}] FALLÓ · {detalle}")
    return await tareas.guardar(
        cliente, tarea.con(estado=cfg.FALLIDA, error=detalle, aprobacion=None)
    )


async def procesar(cliente, grafo, saver, job_id: str) -> None:
    """Ejecuta un trabajo de punta a punta y lo deja en un estado terminal o en pausa."""
    tarea = await tareas.leer(cliente, job_id)
    if tarea is None:
        registrar(f"[{job_id}] descartado: no tiene estado guardado")
        return

    try:
        entrada = await entrada_del_grafo(cliente, saver, tarea)
    except ErrorDeAlmacenamiento as exc:
        await fallar(cliente, tarea, str(exc))
        return

    reanuda = tarea.aprobacion is not None
    registrar(f"[{job_id}] {'reanudando' if reanuda else 'arrancando'} · {tarea.consulta}")
    tarea = await tareas.guardar(
        cliente, tarea.con(estado=cfg.PROCESANDO, aprobacion=None)
    )
    config = config_de(tarea)

    try:
        await correr_tramo(grafo, entrada, config)
    except ErrorDeAgente as exc:
        await fallar(cliente, tarea, str(exc))
        return
    except GraphRecursionError:
        await fallar(cliente, tarea, cfg.ERROR_LIMITE_ORQUESTADOR)
        return
    except Exception as exc:  # último recurso: un worker que muere en silencio deja
        # el trabajo en «processing» para siempre, y nadie se entera de por qué.
        await fallar(cliente, tarea, f"{type(exc).__name__}: {exc}")
        return

    await cerrar(cliente, grafo, tarea, config)


async def consumir(numero: int, cliente, grafo, saver, apagado: asyncio.Event) -> None:
    """Un consumidor: saca trabajos de la cola hasta que se pida el apagado."""
    while not apagado.is_set():
        try:
            job_id = await tareas.tomar(cliente)
        except ErrorDeAlmacenamiento as exc:
            registrar(f"[consumidor {numero}] {exc}")
            await asyncio.sleep(cfg.ESPERA_COLA)
            continue
        if job_id is None:
            continue
        try:
            await procesar(cliente, grafo, saver, job_id)
        except Exception as exc:  # que un trabajo reviente no puede llevarse el consumidor
            registrar(f"[consumidor {numero}] error no previsto en {job_id}: "
                      f"{type(exc).__name__}: {exc}")
        finally:
            # Sale de la cola de proceso haya terminado, pausado o fallado. Si el proceso
            # entero muere no hay `finally`, y ahí el trabajo queda anotado para que lo
            # recupere el arranque siguiente.
            await tareas.soltar(cliente, job_id)


def pedir_apagado(apagado: asyncio.Event) -> None:
    """Engancha SIGINT y SIGTERM para que el worker termine lo que tiene entre manos.

    SIGTERM es la señal con la que Docker frena un contenedor. Sin esto, un `docker compose
    down` cortaría el trabajo en curso a mitad de camino.
    """
    loop = asyncio.get_running_loop()
    for senal in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(senal, apagado.set)
        except NotImplementedError:
            # Windows no implementa add_signal_handler; ahí el Ctrl+C llega como
            # KeyboardInterrupt y lo atrapa `arrancar`.
            signal.signal(senal, lambda *_: apagado.set())


async def arrancar() -> int:
    """Prepara el proceso y corre los consumidores hasta el apagado.

    Lo que bloquea va a un hilo: encender LangSmith hace una petición HTTP sincrónica, y
    construir el índice desde cero embebe el corpus entero contra la API de OpenAI. En el
    event loop, cualquiera de las dos deja al proceso sin atender nada mientras dura.
    """
    await asyncio.to_thread(observability.instrumentar)

    try:
        # Construye el índice si falta, así `git clone && docker compose up` funciona desde
        # cero. Es el worker el que lo hace porque es el único que lo usa.
        vectorstore = await asyncio.to_thread(asegurar_indice)
        await asyncio.to_thread(herramientas.inicializar, vectorstore)
    except ErrorRAG as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    cliente = tareas.conectar()
    apagado = asyncio.Event()
    pedir_apagado(apagado)

    # El saver se crea acá adentro y una sola vez por proceso: `asetup()` captura el event
    # loop en el que se lo llama, y crearlo en otro rompe de una forma difícil de diagnosticar.
    async with tareas.checkpointer() as saver:
        grafo = crear_grafo(checkpointer=saver)
        # Antes de levantar los consumidores: en este instante nada legítimo puede estar en
        # proceso, así que lo que quedó anotado es de un worker que murió.
        for job_id in await tareas.recuperar_huerfanos(cliente):
            registrar(cfg.AVISO_HUERFANO_RECUPERADO.format(job_id=job_id))
        registrar(cfg.AVISO_WORKER_ARRANCA.format(
            consumidores=cfg.CONSUMIDORES, url=obtener_ajustes().redis_url
        ))
        try:
            await asyncio.gather(*(
                consumir(n, cliente, grafo, saver, apagado)
                for n in range(1, cfg.CONSUMIDORES + 1)
            ))
        except KeyboardInterrupt:
            apagado.set()

    registrar(cfg.AVISO_WORKER_APAGA)
    await cliente.aclose()
    await asyncio.to_thread(observability.esperar_trazas)
    return 0


def main() -> int:
    """Punto de entrada del proceso.

    El entorno se carga antes de abrir el event loop: leer el `.env` toca disco, y hacerlo
    adentro sería un bloqueo evitable.
    """
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        cargar_entorno()
    except (ErrorRAG, ErrorDeConfiguracion) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    try:
        return asyncio.run(arrancar())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())

"""La API: acepta consultas, informa su estado y recibe la decisión del revisor humano.

Ninguna ruta corre el grafo. La API escribe intención en Redis y encola; el worker es el único
que ejecuta. La regla es lo que hace que ningún endpoint bloquee: aceptar una consulta y
aprobar una publicación cuestan lo mismo —una escritura y un `RPUSH`— aunque detrás de la
segunda haya varias llamadas al modelo.

Devuelve **202** y no 200 al aceptar trabajo: la tarea fue admitida, no resuelta.

Uso:
    uvicorn app.api.main:app --port 8000
"""

import asyncio
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

import app.nucleo.constantes as cfg
import app.observabilidad.langsmith as observability
import app.trabajos.tareas as tareas
from app.nucleo.errores import ErrorDeAlmacenamiento, ErrorDeConfiguracion, ErrorRAG
from app.nucleo.config import cargar_entorno, obtener_ajustes
from app.nucleo.modelos import modelo_del_rol
from app.grafo.estado import PUBLICAR
from app.trabajos.tareas import Tarea


class ConsultaEntrante(BaseModel):
    """El cuerpo de `POST /tasks`."""

    model_config = ConfigDict(extra="forbid")

    consulta: str = Field(min_length=8, max_length=cfg.LARGO_MAXIMO_CONSULTA)


class VeredictoEntrante(BaseModel):
    """El cuerpo de `POST /tasks/{job_id}/approve`.

    `accion` es una de tres: `publicar` cierra el trabajo, `ampliar` lo manda a buscar más
    doctrina, `rechazar` lo manda a reescribir. Las dos últimas exigen motivos.

    El texto que se juzga sale del estado del grafo y no de este cuerpo: una decisión que
    pudiera declarar sobre qué texto es haría posible aprobar un texto y publicar otro.
    """

    model_config = ConfigDict(extra="forbid")

    accion: Literal["publicar", "ampliar", "rechazar"]
    revisor: str = Field(min_length=1, max_length=80)
    motivos: list[str] = Field(default_factory=list, max_length=cfg.MAXIMO_MOTIVOS)

    def model_post_init(self, _context) -> None:
        if self.accion != PUBLICAR and not self.motivos:
            raise ValueError(cfg.ERROR_DECISION_SIN_MOTIVOS.format(accion=self.accion))
        if any(len(m) > cfg.LARGO_MAXIMO_MOTIVO for m in self.motivos):
            raise ValueError(cfg.ERROR_MOTIVO_LARGO.format(tope=cfg.LARGO_MAXIMO_MOTIVO))


@asynccontextmanager
async def ciclo_de_vida(_app: FastAPI):
    """Abre Redis y el checkpointer al arrancar, y los cierra al apagar.

    El saver se usa acá **solo para leer**: es lo que le da al endpoint de checkpoint acceso
    a lo que escribió el worker desde otro proceso.

    Leer el `.env` y encender LangSmith tocan disco y red de forma sincrónica, así que van a
    un hilo: el arranque no es camino caliente, pero el bloqueo es igual de real.

    Sin configuración la API no levanta, y lo dice en una línea: el traceback de uvicorn
    entierra el mensaje que explica qué falta, que es lo primero que necesita quien recién
    clonó el repositorio.
    """
    try:
        await asyncio.to_thread(cargar_entorno)
    except (ErrorDeConfiguracion, ErrorRAG) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    await asyncio.to_thread(observability.instrumentar)
    cliente = tareas.conectar()
    async with tareas.checkpointer() as saver:
        _app.state.redis = cliente
        _app.state.saver = saver
        yield
    await cliente.aclose()
    await asyncio.to_thread(observability.esperar_trazas)


app = FastAPI(
    title=cfg.TITULO_API,
    description=cfg.DESCRIPCION_API,
    version=cfg.VERSION_API,
    lifespan=ciclo_de_vida,
)


@app.exception_handler(ErrorDeAlmacenamiento)
async def sin_almacenamiento(_request: Request, exc: ErrorDeAlmacenamiento) -> JSONResponse:
    """Traduce cualquier falla de Redis a un 503.

    Es un handler y no un try en cada ruta: sin él, la primera ruta que se olvidara del try
    devolvería un 500 con el traceback de redis-py.
    """
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": str(exc)}
    )


async def buscar(cliente, job_id: str) -> Tarea:
    """La tarea, o un 404 con su id."""
    tarea = await tareas.leer(cliente, job_id)
    if tarea is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            cfg.ERROR_TAREA_INEXISTENTE.format(job_id=job_id),
        )
    return tarea


@app.get("/health")
async def salud(request: Request) -> dict:
    """Si el sistema puede aceptar trabajo, y con qué observabilidad está corriendo."""
    cliente = request.app.state.redis
    ajustes = obtener_ajustes()
    if not await tareas.disponible(cliente):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, cfg.ERROR_REDIS.format(detalle="sin respuesta")
        )
    return {
        "estado": "ok",
        "redis": ajustes.redis_url,
        "trazado": observability.configurada(),
        "proyecto": ajustes.proyecto_langsmith,
        "proveedor": ajustes.llm_provider.value,
        "modelos": {rol: modelo_del_rol(rol, ajustes)
                    for rol in ("supervisor", "investigador", "redactor")},
        "checkpoints": await tareas.contar_checkpoints(cliente),
    }


@app.post("/tasks", status_code=status.HTTP_202_ACCEPTED)
async def crear_tarea(entrante: ConsultaEntrante, request: Request) -> Tarea:
    """Acepta una consulta y la encola. Responde sin esperar a que el grafo corra.

    El estado se persiste **antes** del `RPUSH`: al revés, un worker rápido podría tomar el
    trabajo antes de que exista su estado y descartarlo por inexistente.
    """
    cliente = request.app.state.redis
    job_id = str(uuid.uuid4())
    tarea = Tarea(job_id=job_id, consulta=entrante.consulta, thread_id=job_id)
    await tareas.guardar(cliente, tarea)
    await tareas.encolar(cliente, job_id)
    return tarea


@app.get("/tasks")
async def listar_tareas(request: Request) -> list[Tarea]:
    """Todos los trabajos, del más nuevo al más viejo."""
    return await tareas.listar(request.app.state.redis)


@app.get("/tasks/{job_id}")
async def ver_tarea(job_id: str, request: Request) -> Tarea:
    """El estado de un trabajo. Es lo que consulta quien hace polling."""
    return await buscar(request.app.state.redis, job_id)


@app.post("/tasks/{job_id}/approve", status_code=status.HTTP_202_ACCEPTED)
async def aprobar(job_id: str, veredicto: VeredictoEntrante, request: Request) -> Tarea:
    """Registra la decisión del revisor y devuelve el trabajo a la cola.

    Reanudar es correr el grafo, y en el camino del rechazo son varias llamadas al modelo: por
    eso acá se escribe la decisión y se encola, y reanuda el worker.

    Los dos 409 son distintos. El primero dice que el trabajo no está esperando a nadie —el
    pedido está bien formado, el recurso está en otro estado—; el segundo, que esta pausa ya
    fue decidida, y lo resuelve el `SET NX` de `registrar_decision`.
    """
    cliente = request.app.state.redis
    tarea = await buscar(cliente, job_id)
    if tarea.estado != cfg.ESPERANDO_APROBACION or tarea.aprobacion is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            cfg.ERROR_TAREA_SIN_APROBACION.format(job_id=job_id, estado=tarea.estado),
        )

    primero = await tareas.registrar_decision(
        cliente,
        job_id,
        tarea.aprobacion.interrupt_id,
        {
            "accion": veredicto.accion,
            "revisor": veredicto.revisor,
            "motivos": veredicto.motivos,
            # El reloj lo pone la API, que es donde efectivamente se decidió. Adentro del nodo
            # daría dos valores, porque se re-ejecuta al reanudar.
            "decidida_en": tareas.ahora(),
        },
    )
    if not primero:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            cfg.ERROR_APROBACION_YA_DECIDIDA.format(job_id=job_id),
        )

    # Vuelve a `pending`, que es exactamente lo que es: en la cola, sin nadie trabajándolo. El
    # pedido se conserva porque el `interrupt_id` es lo que le dice al worker qué decisión leer.
    tarea = await tareas.guardar(cliente, tarea.con(estado=cfg.PENDIENTE))
    await tareas.encolar(cliente, job_id)
    return tarea


class Checkpoint(BaseModel):
    """Lo que el checkpointer dejó en Redis para un trabajo."""

    job_id: str
    thread_id: str
    checkpoint_id: str
    guardado_en: str
    escrituras_pendientes: tuple[str, ...]
    pausado: bool
    artefactos: dict
    claves_en_redis: int


@app.get("/tasks/{job_id}/checkpoint")
async def ver_checkpoint(job_id: str, request: Request) -> Checkpoint:
    """El checkpoint del trabajo, leído de Redis desde un proceso que nunca corrió el grafo.

    Existe para que la persistencia sea comprobable y no declarada: lo que devuelve lo escribió
    el worker, y lo lee la API. Si Redis no guardara el estado del grafo, esta ruta no podría
    contestar.
    """
    saver = request.app.state.saver
    tarea = await buscar(request.app.state.redis, job_id)
    tupla = await saver.aget_tuple({"configurable": {"thread_id": tarea.thread_id}})
    if tupla is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, cfg.ERROR_CHECKPOINT_AUSENTE.format(job_id=job_id)
        )

    estado = tupla.checkpoint.get("channel_values", {})
    # Los canales con escritura pendiente. En una pausa aparece `__interrupt__`: es la marca
    # que deja el `interrupt()` y lo que le permite a cualquier worker retomar el trabajo.
    pendientes = tuple(sorted({canal for _, canal, _ in (tupla.pending_writes or [])}))
    return Checkpoint(
        job_id=job_id,
        thread_id=tarea.thread_id,
        checkpoint_id=tupla.checkpoint.get("id", ""),
        guardado_en=tupla.checkpoint.get("ts", ""),
        escrituras_pendientes=pendientes,
        pausado=tarea.estado == cfg.ESPERANDO_APROBACION,
        artefactos={
            "investigaciones": len(estado.get("investigaciones") or ()),
            "verificaciones": len(estado.get("verificaciones") or ()),
            "redacciones": len(estado.get("redacciones") or ()),
            "aprobaciones": len(estado.get("aprobaciones") or ()),
            "mensajes": len(estado.get("messages") or ()),
            "vueltas": estado.get("vueltas", 0),
            "intentos": estado.get("intentos", 0),
        },
        claves_en_redis=await tareas.contar_checkpoints(request.app.state.redis),
    )

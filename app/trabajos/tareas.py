"""El trabajo asincrónico visto desde Redis: su estado, su cola y su decisión humana.

Lo usan la API y el worker, que corren en procesos distintos y no comparten memoria: Redis es
lo único que ambos ven. El modelo vive acá y no en cada uno para que las dos puntas serialicen
y lean la misma forma.

Cada clave tiene un solo escritor, salvo `estado:<job_id>`: la API lo escribe en las dos
transiciones que ocurren mientras ningún worker tiene el trabajo, y el worker en el resto.

Toda falla del cliente sale como `ErrorDeAlmacenamiento`: quien llama traduce eso a un 503 y
no necesita conocer la jerarquía de excepciones de redis-py.
"""

import json
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, Tuple

import redis.asyncio as redis
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from redis.exceptions import RedisError

import app.nucleo.constantes as cfg
from app.nucleo.config import obtener_ajustes
from app.nucleo.errores import ErrorDeAlmacenamiento
from app.grafo.estado import Aprobacion, Cita, Investigacion, Redaccion, Verificacion


def ahora() -> str:
    """El instante actual en ISO 8601 con zona, que es como se guardan las marcas de tiempo."""
    return datetime.now(timezone.utc).isoformat()


class PedidoDeAprobacion(BaseModel):
    """Lo que el grafo dejó sobre la mesa cuando pausó, más la identidad de esa pausa.

    El `interrupt_id` acota la decisión a *esta* pausa: tras un rechazo el grafo vuelve a
    pausar con otro id, y la clave de la decisión anterior no bloquea la nueva.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    interrupt_id: str
    consulta: str
    texto: str
    citas: Tuple[dict, ...] = ()
    sobre_texto: str
    intentos: int = 0
    # Por qué el sistema pidió que lo mire una persona. Es lo primero que lee el revisor.
    motivos: Tuple[str, ...] = ()
    cobertura: float = 0.0
    citas_en_el_texto: int = 0
    # Lo pone el worker al detectar la pausa. Dentro del nodo no puede ir: se re-ejecuta al
    # reanudar y el reloj daría otro valor.
    solicitada_en: str


class Tarea(BaseModel):
    """El estado completo de un trabajo, tal como se guarda en `estado:<job_id>`."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    consulta: str
    estado: str = cfg.PENDIENTE
    # Un hilo por trabajo. Compartirlo entre dos sería un bug concreto: el reducer `acumular`
    # trata la tupla vacía del estado inicial como un no-op, así que los artefactos del
    # trabajo anterior sobrevivirían al arranque del siguiente.
    thread_id: str
    creada_en: str = Field(default_factory=ahora)
    actualizada_en: str = Field(default_factory=ahora)
    resultado: Optional[str] = None
    citas: Tuple[str, ...] = ()
    publicado: bool = False
    revisiones: int = 0
    aprobacion: Optional[PedidoDeAprobacion] = None
    error: Optional[str] = None

    def con(self, **cambios) -> "Tarea":
        """Una copia con los campos cambiados y la marca de tiempo al día."""
        return self.model_copy(update={**cambios, "actualizada_en": ahora()})


def conectar(url: Optional[str] = None) -> redis.Redis:
    """Abre el cliente asincrónico contra Redis.

    `decode_responses` evita que cada lectura tenga que decodificar bytes a mano.
    """
    return redis.from_url(url or obtener_ajustes().redis_url, decode_responses=True)


async def disponible(cliente: redis.Redis) -> bool:
    """Si Redis responde. Es lo que mira `/health`."""
    try:
        return bool(await cliente.ping())
    except RedisError:
        return False


async def guardar(cliente: redis.Redis, tarea: Tarea) -> Tarea:
    """Persiste el estado del trabajo. Sin TTL, a propósito.

    Un trabajo detenido en `awaiting_approval` puede esperar a una persona todo lo que haga
    falta; si su estado venciera, la reanudación fallaría y el trabajo se perdería en silencio.
    """
    try:
        await cliente.set(
            cfg.CLAVE_ESTADO.format(job_id=tarea.job_id), tarea.model_dump_json()
        )
    except RedisError as exc:
        raise ErrorDeAlmacenamiento(cfg.ERROR_REDIS.format(detalle=exc)) from exc
    return tarea


async def leer(cliente: redis.Redis, job_id: str) -> Optional[Tarea]:
    """El estado del trabajo, o None si no existe."""
    try:
        crudo = await cliente.get(cfg.CLAVE_ESTADO.format(job_id=job_id))
    except RedisError as exc:
        raise ErrorDeAlmacenamiento(cfg.ERROR_REDIS.format(detalle=exc)) from exc
    if crudo is None:
        return None
    try:
        return Tarea.model_validate_json(crudo)
    except ValidationError as exc:
        raise ErrorDeAlmacenamiento(
            cfg.ERROR_ESTADO_ILEGIBLE.format(job_id=job_id, detalle=exc.error_count())
        ) from exc


async def listar(cliente: redis.Redis) -> list[Tarea]:
    """Todos los trabajos legibles, del más nuevo al más viejo.

    Recorre con SCAN y no con KEYS: KEYS bloquea el servidor mientras barre el espacio de
    claves entero.

    Un estado que no se puede leer se saltea y se avisa por stderr. Es la diferencia con
    `leer`, que sí falla: un trabajo roto no puede dejar sin listado a los otros doscientos,
    y quien quiera saber qué le pasó a ese lo consulta por su id y recibe el error.
    """
    tareas, ilegibles = [], []
    try:
        async for clave in cliente.scan_iter(match=cfg.CLAVE_ESTADO.format(job_id="*")):
            crudo = await cliente.get(clave)
            if not crudo:
                continue
            try:
                tareas.append(Tarea.model_validate_json(crudo))
            except ValidationError:
                ilegibles.append(clave)
    except RedisError as exc:
        raise ErrorDeAlmacenamiento(cfg.ERROR_REDIS.format(detalle=exc)) from exc
    if ilegibles:
        print(cfg.AVISO_ESTADOS_ILEGIBLES.format(claves=", ".join(ilegibles)), file=sys.stderr)
    return sorted(tareas, key=lambda t: t.creada_en, reverse=True)


async def encolar(cliente: redis.Redis, job_id: str) -> None:
    """Pone el trabajo en la cola. Solo la API escribe acá."""
    try:
        await cliente.rpush(cfg.COLA_TAREAS, job_id)
    except RedisError as exc:
        raise ErrorDeAlmacenamiento(cfg.ERROR_REDIS.format(detalle=exc)) from exc


async def tomar(cliente: redis.Redis, espera: int = cfg.ESPERA_COLA) -> Optional[str]:
    """Mueve el próximo trabajo a la cola de proceso y lo devuelve.

    `BLMOVE` y no `BLPOP`: el id sigue anotado en `cola:procesando` mientras se trabaja, y eso
    es lo que hace recuperable a un worker que muere de golpe. Con `BLPOP`, entre sacarlo de
    la cola y llegar a un estado terminal el trabajo no estaba en ningún lado desde donde
    retomarlo.

    Bloquea hasta `espera` segundos; el timeout existe para que el apagado se note sin tener
    que matar la corrutina.

    Verificado contra la documentación oficial: `BLMOVE` mueve el elemento entre las dos
    listas de forma atómica. https://redis.io/docs/latest/commands/blmove/
    """
    try:
        return await cliente.blmove(
            cfg.COLA_TAREAS, cfg.COLA_PROCESANDO, espera, src="LEFT", dest="RIGHT"
        )
    except RedisError as exc:
        raise ErrorDeAlmacenamiento(cfg.ERROR_REDIS.format(detalle=exc)) from exc


async def soltar(cliente: redis.Redis, job_id: str) -> None:
    """Saca el trabajo de la cola de proceso: ya nadie lo tiene entre manos.

    Se llama en el `finally` del consumidor, así vale igual si el trabajo terminó, quedó
    esperando a un revisor o falló.
    """
    try:
        await cliente.lrem(cfg.COLA_PROCESANDO, 1, job_id)
    except RedisError as exc:
        raise ErrorDeAlmacenamiento(cfg.ERROR_REDIS.format(detalle=exc)) from exc


async def recuperar_huerfanos(cliente: redis.Redis) -> tuple[str, ...]:
    """Devuelve a la cola los trabajos que un worker anterior dejó en proceso.

    Corre una sola vez al arrancar, antes de que exista ningún consumidor: en ese instante
    nada legítimo puede estar en proceso, así que lo que haya ahí quedó de un proceso que
    murió. Vuelven al frente de la cola porque ya habían empezado a esperar.
    """
    recuperados: list[str] = []
    try:
        while True:
            job_id = await cliente.lmove(
                cfg.COLA_PROCESANDO, cfg.COLA_TAREAS, src="LEFT", dest="LEFT"
            )
            if job_id is None:
                break
            recuperados.append(job_id)
    except RedisError as exc:
        raise ErrorDeAlmacenamiento(cfg.ERROR_REDIS.format(detalle=exc)) from exc
    return tuple(recuperados)


async def registrar_decision(
    cliente: redis.Redis, job_id: str, interrupt_id: str, veredicto: dict
) -> bool:
    """Guarda el veredicto humano y devuelve si fue el primero.

    El `NX` es el control de la doble aprobación: dos revisores que decidan a la vez sobre la
    misma pausa producen una sola escritura, y el segundo se entera de que llegó tarde.
    """
    try:
        escrito = await cliente.set(
            cfg.CLAVE_DECISION.format(job_id=job_id, interrupt_id=interrupt_id),
            json.dumps(veredicto),
            nx=True,
        )
    except RedisError as exc:
        raise ErrorDeAlmacenamiento(cfg.ERROR_REDIS.format(detalle=exc)) from exc
    return bool(escrito)


async def leer_decision(
    cliente: redis.Redis, job_id: str, interrupt_id: str
) -> Optional[dict]:
    """El veredicto que dejó la API, que es lo que el worker le pasa al grafo al reanudar."""
    try:
        crudo = await cliente.get(
            cfg.CLAVE_DECISION.format(job_id=job_id, interrupt_id=interrupt_id)
        )
    except RedisError as exc:
        raise ErrorDeAlmacenamiento(cfg.ERROR_REDIS.format(detalle=exc)) from exc
    return json.loads(crudo) if crudo else None


async def contar_checkpoints(cliente: redis.Redis) -> int:
    """Cuántas claves escribió el checkpointer. Es lo que hace visible el 25 % de Redis."""
    total = 0
    try:
        async for _ in cliente.scan_iter(match=cfg.PATRON_CHECKPOINT, count=500):
            total += 1
    except RedisError as exc:
        raise ErrorDeAlmacenamiento(cfg.ERROR_REDIS.format(detalle=exc)) from exc
    return total


# Los artefactos que viajan al checkpoint. El serde de Redis guarda en JSON —para que
# RedisJSON pueda indexarlo— y al leer reconstruye solo los tipos de una allowlist; sin
# registrarlos, `Investigacion` y `Redaccion` vuelven como diccionarios y el primer `.texto`
# revienta con un AttributeError que no menciona a Redis. La lista se deriva de las clases:
# escrita a mano, un renombre la dejaría apuntando a un símbolo que ya no existe.
MODELOS_PERSISTIDOS = tuple(
    tuple(modelo.__module__.split(".")) + (modelo.__name__,)
    for modelo in (Cita, Investigacion, Verificacion, Redaccion, Aprobacion)
)


@asynccontextmanager
async def checkpointer(url: Optional[str] = None):
    """El checkpointer de LangGraph, con los artefactos del proyecto ya registrados.

    Es la única puerta a los checkpoints: el worker lo usa para escribir y la API para leer, y
    los dos necesitan la misma allowlist para ver los mismos objetos.

    `asetup()` crea los índices que el checkpointer necesita. Es idempotente y seguro entre
    procesos, así que lo llaman los dos sin coordinarse.
    """
    async with AsyncRedisSaver.from_conn_string(url or obtener_ajustes().redis_url) as saver:
        saver.serde = JsonPlusRedisSerializer(allowed_json_modules=MODELOS_PERSISTIDOS)
        await saver.asetup()
        yield saver

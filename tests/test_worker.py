"""La cola de trabajos y el consumidor, contra un Redis en memoria.

El foco es lo que pasa cuando un worker muere con un trabajo entre manos: el id tiene que
quedar anotado en la cola de proceso y volver a la cola al arrancar el siguiente. Con `BLPOP`
ese trabajo desaparecía y quedaba en `processing` para siempre.

El checkpointer real necesita RediSearch y RedisJSON, que un doble no reproduce: eso se
prueba contra el servicio vivo, en `demo.py`.
"""

import pytest
from langgraph.types import Command

from app.nucleo import constantes as cfg
from app.nucleo.errores import ErrorDeAlmacenamiento
from app.trabajos import tareas, worker
from app.trabajos.tareas import PedidoDeAprobacion, Tarea


def tarea_nueva(job_id: str = "job-1") -> Tarea:
    return Tarea(job_id=job_id, consulta="¿Que es el exceso ritual manifiesto?",
                 thread_id=job_id)


def pedido(interrupt_id: str = "int-1") -> PedidoDeAprobacion:
    return PedidoDeAprobacion(
        interrupt_id=interrupt_id, consulta="¿Que es el exceso ritual manifiesto?",
        texto="Un borrador con dos citas.", citas=(), sobre_texto="huella123",
        intentos=0, motivos=("pocas citas",), cobertura=1.0, citas_en_el_texto=2,
        solicitada_en=tareas.ahora())


class SaverFalso:
    """Devuelve la tupla de checkpoint que se le configure."""

    def __init__(self, tupla=None):
        self.tupla = tupla

    async def aget_tuple(self, _config):
        return self.tupla


class TestColaDeProceso:
    """El trabajo queda anotado mientras se procesa."""

    async def test_tomar_mueve_el_trabajo_a_la_cola_de_proceso(self, redis_falso):
        await tareas.encolar(redis_falso, "job-1")
        job_id = await tareas.tomar(redis_falso, espera=0)
        assert job_id == "job-1"
        assert redis_falso.listas[cfg.COLA_TAREAS] == []
        assert redis_falso.listas[cfg.COLA_PROCESANDO] == ["job-1"]

    async def test_soltar_lo_saca_de_la_cola_de_proceso(self, redis_falso):
        await tareas.encolar(redis_falso, "job-1")
        await tareas.tomar(redis_falso, espera=0)
        await tareas.soltar(redis_falso, "job-1")
        assert redis_falso.listas[cfg.COLA_PROCESANDO] == []

    async def test_con_la_cola_vacia_devuelve_none(self, redis_falso):
        assert await tareas.tomar(redis_falso, espera=0) is None

    async def test_el_orden_de_la_cola_se_respeta(self, redis_falso):
        for job_id in ("job-1", "job-2", "job-3"):
            await tareas.encolar(redis_falso, job_id)
        assert await tareas.tomar(redis_falso, espera=0) == "job-1"
        assert await tareas.tomar(redis_falso, espera=0) == "job-2"

    async def test_una_caida_de_redis_sale_como_error_de_almacenamiento(self, redis_falso):
        redis_falso.caido = True
        with pytest.raises(ErrorDeAlmacenamiento):
            await tareas.tomar(redis_falso, espera=0)


class TestRecuperarHuerfanos:
    """Lo que un worker muerto dejó a medias vuelve a la cola."""

    async def test_un_trabajo_tomado_y_no_soltado_se_recupera(self, redis_falso):
        # La muerte del proceso: se tomó el trabajo y nunca corrió el `finally`.
        await tareas.encolar(redis_falso, "job-1")
        await tareas.tomar(redis_falso, espera=0)
        assert redis_falso.listas[cfg.COLA_PROCESANDO] == ["job-1"]

        recuperados = await tareas.recuperar_huerfanos(redis_falso)
        assert recuperados == ("job-1",)
        assert redis_falso.listas[cfg.COLA_TAREAS] == ["job-1"]
        assert redis_falso.listas[cfg.COLA_PROCESANDO] == []

    async def test_los_recuperados_van_al_frente_de_la_cola(self, redis_falso):
        # Ya habían empezado a esperar: van antes que los que llegaron después.
        await tareas.encolar(redis_falso, "viejo")
        await tareas.tomar(redis_falso, espera=0)
        await tareas.encolar(redis_falso, "nuevo")

        await tareas.recuperar_huerfanos(redis_falso)
        assert redis_falso.listas[cfg.COLA_TAREAS] == ["viejo", "nuevo"]

    async def test_sin_huerfanos_no_recupera_nada(self, redis_falso):
        assert await tareas.recuperar_huerfanos(redis_falso) == ()

    async def test_recupera_varios(self, redis_falso):
        for job_id in ("job-1", "job-2"):
            await tareas.encolar(redis_falso, job_id)
            await tareas.tomar(redis_falso, espera=0)
        assert set(await tareas.recuperar_huerfanos(redis_falso)) == {"job-1", "job-2"}


class TestEntradaDelGrafo:
    """Los tres caminos según cómo llegó el trabajo a la cola."""

    async def test_un_trabajo_nuevo_arranca_de_cero(self, redis_falso):
        entrada = await worker.entrada_del_grafo(redis_falso, SaverFalso(), tarea_nueva())
        assert entrada["consulta"] == "¿Que es el exceso ritual manifiesto?"
        assert entrada["vueltas"] == 0

    async def test_un_huerfano_con_checkpoint_reanuda_sin_pisar_el_estado(self, redis_falso):
        # `None` y no el estado inicial: reinyectarlo pondría `vueltas` e `intentos` en cero,
        # y esos campos no tienen reducer que los proteja.
        entrada = await worker.entrada_del_grafo(
            redis_falso, SaverFalso(tupla=object()), tarea_nueva())
        assert entrada is None

    async def test_un_veredicto_humano_reanuda_con_command(self, redis_falso):
        tarea = tarea_nueva().con(estado=cfg.ESPERANDO_APROBACION, aprobacion=pedido())
        await tareas.registrar_decision(redis_falso, "job-1", "int-1", {"accion": "publicar"})
        entrada = await worker.entrada_del_grafo(redis_falso, SaverFalso(), tarea)
        assert isinstance(entrada, Command)
        assert entrada.resume == {"accion": "publicar"}

    async def test_una_pausa_sin_decision_es_un_error(self, redis_falso):
        # Un trabajo en espera vuelve a la cola recién cuando la API escribió la decisión:
        # sin ella, arrancar de cero perdería lo hecho.
        tarea = tarea_nueva().con(estado=cfg.ESPERANDO_APROBACION, aprobacion=pedido())
        with pytest.raises(ErrorDeAlmacenamiento):
            await worker.entrada_del_grafo(redis_falso, SaverFalso(), tarea)


class TestProcesar:
    """El ciclo de un trabajo, con un grafo de mentira."""

    class GrafoFalso:
        """Corre sin hacer nada y devuelve el snapshot que se le configure."""

        def __init__(self, snapshot, explota=None):
            self.snapshot, self.explota = snapshot, explota

        async def astream(self, *_a, **_k):
            if self.explota:
                raise self.explota
            for nada in ():
                yield nada

        async def aget_state(self, _config):
            return self.snapshot

    class Snapshot:
        def __init__(self, values, interrupts=()):
            self.values, self.interrupts = values, interrupts

    async def test_un_trabajo_sin_estado_se_descarta(self, redis_falso):
        # Encolado sin su estado: la API escribe el estado antes de encolar, así que esto
        # solo pasa con un id inventado.
        await worker.procesar(redis_falso, self.GrafoFalso(None), SaverFalso(), "inventado")
        assert redis_falso.claves == {}

    async def test_un_trabajo_completo_queda_en_completed(self, redis_falso):
        await tareas.guardar(redis_falso, tarea_nueva())
        snapshot = self.Snapshot({"redacciones": (), "publicado": True, "aprobaciones": ()})
        await worker.procesar(redis_falso, self.GrafoFalso(snapshot), SaverFalso(), "job-1")
        tarea = await tareas.leer(redis_falso, "job-1")
        assert tarea.estado == cfg.COMPLETADA
        assert tarea.publicado is True

    async def test_un_error_deja_el_trabajo_en_failed_con_su_motivo(self, redis_falso):
        from app.nucleo.errores import ErrorDeAgente

        await tareas.guardar(redis_falso, tarea_nueva())
        grafo = self.GrafoFalso(None, explota=ErrorDeAgente("el modelo no respondio"))
        await worker.procesar(redis_falso, grafo, SaverFalso(), "job-1")
        tarea = await tareas.leer(redis_falso, "job-1")
        assert tarea.estado == cfg.FALLIDA
        assert "no respondio" in tarea.error

    async def test_una_pausa_deja_el_trabajo_esperando(self, redis_falso):
        class Pausa:
            id = "int-1"
            value = {"consulta": "¿Que es el exceso ritual manifiesto?",
                     "texto": "Un borrador.", "citas": (), "sobre_texto": "huella123",
                     "intentos": 0, "motivos": ("pocas citas",), "cobertura": 1.0,
                     "citas_en_el_texto": 2}

        await tareas.guardar(redis_falso, tarea_nueva())
        snapshot = self.Snapshot({"aprobaciones": ()}, interrupts=(Pausa(),))
        await worker.procesar(redis_falso, self.GrafoFalso(snapshot), SaverFalso(), "job-1")
        tarea = await tareas.leer(redis_falso, "job-1")
        assert tarea.estado == cfg.ESPERANDO_APROBACION
        assert tarea.aprobacion.interrupt_id == "int-1"

    async def test_una_pausa_ya_decidida_vuelve_a_la_cola_en_vez_de_bloquearse(self, redis_falso):
        # El rincón del crash entre la decisión y el final de la reanudación: dejarlo en
        # `awaiting_approval` lo bloquearía, porque el SET NX ya está tomado.
        class Pausa:
            id = "int-1"
            value = {"consulta": "¿Que es el exceso ritual manifiesto?",
                     "texto": "Un borrador.", "citas": (), "sobre_texto": "huella123",
                     "intentos": 0, "motivos": ("pocas citas",), "cobertura": 1.0,
                     "citas_en_el_texto": 2}

        await tareas.guardar(redis_falso, tarea_nueva())
        await tareas.registrar_decision(redis_falso, "job-1", "int-1", {"accion": "publicar"})
        snapshot = self.Snapshot({"aprobaciones": ()}, interrupts=(Pausa(),))
        await worker.procesar(redis_falso, self.GrafoFalso(snapshot), SaverFalso(), "job-1")

        tarea = await tareas.leer(redis_falso, "job-1")
        assert tarea.estado == cfg.PENDIENTE
        assert "job-1" in redis_falso.listas[cfg.COLA_TAREAS]

"""Los seis endpoints, contra un Redis en memoria.

`ASGITransport` habla con la aplicación sin abrir un puerto y sin correr el lifespan, así que
`app.state.redis` y `app.state.saver` se inyectan a mano: es el mismo hueco por donde el
lifespan pone los reales. La suite no necesita Redis ni el checkpointer de RedisJSON.

Cada código de estado se comprueba con su causa: un 409 por estado equivocado y otro por
decisión repetida son fallas distintas para quien consume la API.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.main import app
from app.nucleo import constantes as cfg
from app.nucleo.errores import ErrorDeAlmacenamiento
from app.trabajos import tareas
from app.trabajos.tareas import PedidoDeAprobacion, Tarea


@pytest.fixture
async def api(redis_falso):
    """Un cliente HTTP contra la app, con Redis y saver de mentira ya puestos.

    ASGITransport no corre el lifespan, así que `app.state` se llena a mano: es el mismo
    hueco por donde el ciclo de vida real pone el cliente y el checkpointer.
    """
    app.state.redis = redis_falso
    app.state.saver = SaverFalso()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as cliente:
        yield cliente


class SaverFalso:
    """Devuelve el checkpoint que se le configure, o ninguno."""

    def __init__(self, tupla=None):
        self.tupla = tupla

    async def aget_tuple(self, _config):
        return self.tupla


async def encolar_una(api, consulta="¿Que es el exceso ritual manifiesto?") -> dict:
    """Crea un trabajo por la API y devuelve su representación."""
    respuesta = await api.post("/tasks", json={"consulta": consulta})
    return respuesta.json()


class TestCrearTarea:
    """POST /tasks: acepta y encola, sin correr el grafo."""

    async def test_devuelve_202_con_el_trabajo_pendiente(self, api):
        r = await api.post("/tasks", json={"consulta": "¿Que es el exceso ritual?"})
        assert r.status_code == 202
        cuerpo = r.json()
        assert cuerpo["estado"] == cfg.PENDIENTE
        assert cuerpo["job_id"] == cuerpo["thread_id"]      # un hilo por trabajo

    async def test_el_estado_se_escribe_antes_de_encolar(self, api, redis_falso):
        # Al revés, un worker rápido tomaría el trabajo antes de que exista su estado y lo
        # descartaría por inexistente.
        await encolar_una(api)
        verbos = [t[0] for t in redis_falso.traza]
        assert verbos.index("set") < verbos.index("rpush")

    async def test_una_consulta_corta_no_pasa_la_validacion(self, api):
        r = await api.post("/tasks", json={"consulta": "corta"})
        assert r.status_code == 422

    async def test_una_consulta_larguisima_no_pasa(self, api):
        r = await api.post("/tasks", json={"consulta": "x" * (cfg.LARGO_MAXIMO_CONSULTA + 1)})
        assert r.status_code == 422


class TestConsultas:
    """GET /tasks y GET /tasks/{job_id}."""

    async def test_un_trabajo_inexistente_da_404(self, api):
        r = await api.get("/tasks/no-existe")
        assert r.status_code == 404

    async def test_el_trabajo_creado_se_puede_leer(self, api):
        creada = await encolar_una(api)
        r = await api.get(f"/tasks/{creada['job_id']}")
        assert r.status_code == 200
        assert r.json()["job_id"] == creada["job_id"]

    async def test_el_listado_trae_los_trabajos(self, api):
        await encolar_una(api, "primera consulta sobre arbitrariedad")
        await encolar_una(api, "segunda consulta sobre el per saltum")
        r = await api.get("/tasks")
        assert r.status_code == 200
        assert len(r.json()) == 2


class TestAprobacion:
    """POST /tasks/{job_id}/approve: la señal externa que reanuda el grafo."""

    @pytest.fixture
    def esperando(self, redis_falso):
        """Un trabajo detenido esperando a un revisor."""
        pedido = PedidoDeAprobacion(
            interrupt_id="int-1", consulta="¿Que es el exceso ritual manifiesto?",
            texto="Un borrador con dos citas.", citas=(), sobre_texto="huella123",
            intentos=0, motivos=("pocas citas",), cobertura=1.0, citas_en_el_texto=2,
            solicitada_en=tareas.ahora())
        tarea = Tarea(job_id="job-1", consulta="¿Que es el exceso ritual manifiesto?",
                      thread_id="job-1", estado=cfg.ESPERANDO_APROBACION, aprobacion=pedido)
        redis_falso.claves[cfg.CLAVE_ESTADO.format(job_id="job-1")] = tarea.model_dump_json()
        return tarea

    async def test_publicar_devuelve_el_trabajo_a_la_cola(self, api, esperando, redis_falso):
        r = await api.post("/tasks/job-1/approve",
                               json={"accion": "publicar", "revisor": "felipe", "motivos": []})
        assert r.status_code == 202
        assert r.json()["estado"] == cfg.PENDIENTE
        assert "job-1" in redis_falso.listas[cfg.COLA_TAREAS]

    async def test_un_trabajo_que_no_espera_nada_da_409(self, api):
        creada = await encolar_una(api)
        r = await api.post(f"/tasks/{creada['job_id']}/approve",
                               json={"accion": "publicar", "revisor": "felipe", "motivos": []})
        assert r.status_code == 409

    async def test_la_segunda_decision_sobre_la_misma_pausa_da_409(self, api, esperando):
        # Es el SET NX: dos revisores que aprueban a la vez no pueden reanudar dos veces.
        primera = await api.post(
            "/tasks/job-1/approve",
            json={"accion": "publicar", "revisor": "felipe", "motivos": []})
        segunda = await api.post(
            "/tasks/job-1/approve",
            json={"accion": "rechazar", "revisor": "otro", "motivos": ["no me gusta"]})
        assert primera.status_code == 202
        assert segunda.status_code == 409

    async def test_una_accion_inventada_no_pasa(self, api, esperando):
        r = await api.post("/tasks/job-1/approve",
                               json={"accion": "archivar", "revisor": "felipe", "motivos": ["x"]})
        assert r.status_code == 422

    async def test_rechazar_sin_motivos_no_pasa(self, api, esperando):
        # Sin decir qué falta, "reescribí" es una instrucción vacía.
        r = await api.post("/tasks/job-1/approve",
                               json={"accion": "rechazar", "revisor": "felipe", "motivos": []})
        assert r.status_code == 422

    async def test_un_motivo_larguisimo_no_pasa(self, api, esperando):
        r = await api.post(
            "/tasks/job-1/approve",
            json={"accion": "rechazar", "revisor": "felipe",
                  "motivos": ["x" * (cfg.LARGO_MAXIMO_MOTIVO + 1)]})
        assert r.status_code == 422

    async def test_aprobar_un_trabajo_inexistente_da_404(self, api):
        r = await api.post("/tasks/no-existe/approve",
                               json={"accion": "publicar", "revisor": "felipe", "motivos": []})
        assert r.status_code == 404


class TestCheckpoint:
    """GET /tasks/{job_id}/checkpoint: la persistencia, leída desde otro proceso."""

    async def test_sin_checkpoint_todavia_da_404(self, api):
        creada = await encolar_una(api)
        r = await api.get(f"/tasks/{creada['job_id']}/checkpoint")
        assert r.status_code == 404

    async def test_devuelve_el_contenido_del_checkpoint(self, api, redis_falso, investigacion):
        # Un trabajo detenido: `pausado` sale de su estado, que es lo que ve quien consulta.
        pedido = PedidoDeAprobacion(
            interrupt_id="int-1", consulta="¿Que es el exceso ritual manifiesto?",
            texto="Un borrador con dos citas.", citas=(), sobre_texto="huella123",
            intentos=0, motivos=("pocas citas",), cobertura=1.0, citas_en_el_texto=2,
            solicitada_en=tareas.ahora())
        tarea = Tarea(job_id="job-2", consulta="¿Que es el exceso ritual manifiesto?",
                      thread_id="job-2", estado=cfg.ESPERANDO_APROBACION, aprobacion=pedido)
        redis_falso.claves[cfg.CLAVE_ESTADO.format(job_id="job-2")] = tarea.model_dump_json()

        class TuplaFalsa:
            checkpoint = {"id": "chk-1", "ts": "2026-09-01T12:00:00+00:00",
                          "channel_values": {"investigaciones": (investigacion,),
                                             "verificaciones": (), "redacciones": (),
                                             "aprobaciones": (), "vueltas": 2, "intentos": 0}}
            pending_writes = [("tarea-1", "__interrupt__", None)]

        app.state.saver = SaverFalso(TuplaFalsa())
        r = await api.get("/tasks/job-2/checkpoint")
        assert r.status_code == 200
        cuerpo = r.json()
        assert cuerpo["pausado"] is True
        assert "__interrupt__" in cuerpo["escrituras_pendientes"]
        assert cuerpo["artefactos"]["investigaciones"] == 1


class TestSalud:
    """GET /health y la traducción de una caída de Redis."""

    async def test_con_redis_arriba_responde_ok(self, api):
        r = await api.get("/health")
        assert r.status_code == 200
        assert r.json()["estado"] == "ok"

    async def test_con_redis_caido_responde_503(self, api, redis_falso):
        redis_falso.caido = True
        r = await api.get("/health")
        assert r.status_code == 503

    async def test_una_falla_de_almacenamiento_sale_como_503_y_no_como_500(self, api, monkeypatch):
        # El handler existe para que una ruta que se olvide del try no devuelva el traceback
        # de redis-py con un 500.
        async def explota(*_args, **_kwargs):
            raise ErrorDeAlmacenamiento("redis no responde")

        monkeypatch.setattr(tareas, "listar", explota)
        r = await api.get("/tasks")
        assert r.status_code == 503

"""El grafo real corriendo con dobles: ruteo, pausa, reanudación y frenos.

Lo que se reemplaza son los tres especialistas y el supervisor; el ruteo, el nodo de
publicación y el checkpointer son los que están bajo prueba. Con `InMemorySaver` cada
comprobación tarda milisegundos y no gasta un token.
"""

import pytest
from langgraph.types import Command

from app.grafo import construccion
from app.grafo.estado import AMPLIAR, PUBLICAR, RECHAZAR, leer_situacion
from app.nucleo import constantes as cfg
from app.nucleo.errores import ErrorDeAgente
from tests.dobles import (
    armar,
    estado_inicial,
    pasadas,
    redactor_escaso,
    redactor_sucio,
    supervisor_con_freno,
    veredicto,
)

CONFIG = {"configurable": {"thread_id": "test"}, "recursion_limit": cfg.LIMITE_RECURSION_ORQUESTADOR}


def config(hilo: str) -> dict:
    return {"configurable": {"thread_id": hilo},
            "recursion_limit": cfg.LIMITE_RECURSION_ORQUESTADOR}


class TestCaminoFeliz:
    """Un trabajo sin señales de alerta se publica sin molestar a nadie."""

    async def test_el_trabajo_holgado_se_publica_solo(self):
        grafo = armar()
        final = await grafo.ainvoke(estado_inicial(), config("feliz"))
        assert final["publicado"] is True
        assert final["aprobaciones"] == ()          # nadie tuvo que decidir
        assert len(final["redacciones"]) == 1

    async def test_el_estado_acumula_los_artefactos_de_cada_etapa(self):
        grafo = armar()
        final = await grafo.ainvoke(estado_inicial(), config("acumula"))
        assert len(final["investigaciones"]) == 1
        assert len(final["verificaciones"]) == 1
        assert leer_situacion(final).listo is True


class TestPausaHumana:
    """Un trabajo flojo se detiene y muestra qué lo hizo dudar."""

    async def test_el_trabajo_escaso_se_detiene_con_sus_motivos(self):
        grafo = armar(redactor=redactor_escaso)
        salida = await grafo.ainvoke(estado_inicial(), config("pausa"))
        assert "__interrupt__" in salida
        pedido = salida["__interrupt__"][0].value
        assert pedido["citas_en_el_texto"] < cfg.CITAS_HOLGADAS
        assert pedido["motivos"]
        assert pedido["sobre_texto"]                # la huella del texto que se juzga

    async def test_publicar_cierra_el_grafo(self):
        grafo = armar(redactor=redactor_escaso)
        await grafo.ainvoke(estado_inicial(), config("publicar"))
        final = await grafo.ainvoke(Command(resume=veredicto(PUBLICAR)), config("publicar"))
        assert final["publicado"] is True
        assert len(final["aprobaciones"]) == 1
        assert final["aprobaciones"][-1].accion == PUBLICAR

    async def test_rechazar_vuelve_al_redactor(self):
        grafo = armar(redactor=redactor_escaso)
        await grafo.ainvoke(estado_inicial(), config("rechazar"))
        final = await grafo.ainvoke(
            Command(resume=veredicto(RECHAZAR, ("falta desarrollo",))), config("rechazar"))
        assert len(final["investigaciones"]) == 1   # el material no se rehace
        assert len(final["redacciones"]) == 2       # el texto sí
        assert final["intentos"] >= 1

    async def test_ampliar_vuelve_al_investigador_y_trae_mas_material(self):
        grafo = armar(redactor=redactor_escaso)
        await grafo.ainvoke(estado_inicial(), config("ampliar"))
        final = await grafo.ainvoke(
            Command(resume=veredicto(AMPLIAR, ("poca doctrina",))), config("ampliar"))
        assert len(final["investigaciones"]) == 2
        citas_antes = len(final["investigaciones"][0].citas)
        citas_despues = len(final["investigaciones"][1].citas)
        assert citas_despues > citas_antes
        assert final["aprobaciones"][-1].accion == AMPLIAR


class TestFrenos:
    """Los topes cierran el trabajo sin pedirle nada a nadie."""

    async def test_el_cierre_por_freno_no_pausa(self):
        # Un trabajo que agotó sus intentos cierra con lo que tiene: pedir una revisión
        # sobre un trabajo que ya no se puede corregir sería un peaje inútil.
        grafo = armar(redactor=redactor_sucio, supervisor=supervisor_con_freno)
        final = await grafo.ainvoke(estado_inicial(), config("freno"))
        assert "__interrupt__" not in final
        assert final["publicado"] is False
        assert leer_situacion(final).listo is False


class TestIdempotencia:
    """El nodo de publicación se re-ejecuta entero al reanudar."""

    async def test_el_tramo_previo_al_interrupt_corre_dos_veces(self):
        grafo = armar(redactor=redactor_escaso, contar=True)
        await grafo.ainvoke(estado_inicial(), config("idem"))
        assert (pasadas["antes"], pasadas["despues"]) == (1, 0)

        await grafo.ainvoke(Command(resume=veredicto(PUBLICAR)), config("idem"))
        # Por eso nada con efecto puede ir arriba del interrupt().
        assert (pasadas["antes"], pasadas["despues"]) == (2, 1)


class TestPersistencia:
    """El checkpointer guarda el recorrido, no solo el resultado."""

    async def test_el_checkpoint_conserva_el_estado_de_la_pausa(self):
        grafo = armar(redactor=redactor_escaso)
        await grafo.ainvoke(estado_inicial(), config("checkpoint"))
        snapshot = await grafo.aget_state(config("checkpoint"))
        assert snapshot.interrupts                      # el trabajo quedó esperando
        assert snapshot.values["redacciones"]
        assert snapshot.values["publicado"] is False

    async def test_dos_hilos_no_se_mezclan(self):
        grafo = armar()
        await grafo.ainvoke(estado_inicial("primera consulta de prueba"), config("hilo-a"))
        await grafo.ainvoke(estado_inicial("segunda consulta de prueba"), config("hilo-b"))
        a = await grafo.aget_state(config("hilo-a"))
        b = await grafo.aget_state(config("hilo-b"))
        assert a.values["consulta"] != b.values["consulta"]


class TestNodosInyectables:
    """Los cinco nodos entran por parámetro, y ninguno se reemplaza mutando el módulo.

    Antes el supervisor era el único que no era parámetro, así que los dobles tenían que
    reasignar el símbolo del módulo: la mutación persistía entre tests y no alcanzaba a quien
    hubiera importado el nodo directo.
    """

    def test_los_cinco_nodos_son_parametros(self):
        import inspect

        from app.grafo.estado import NODOS

        firma = inspect.signature(construccion.crear_grafo)
        esperados = set(NODOS) | {"supervisor", "aprobacion"}
        assert esperados <= set(firma.parameters)

    def test_cada_parametro_trae_la_implementacion_real(self):
        import inspect

        from app.agentes import analyst_agent, research_agent, writer_agent
        from app.grafo import hitl, supervisor as sup

        reales = {
            "investigador": research_agent.investigador_node,
            "verificador": analyst_agent.verificador_node,
            "redactor": writer_agent.redactor_node,
            "supervisor": sup.supervisor_node,
            "aprobacion": hitl.aprobacion_node,
        }
        firma = inspect.signature(construccion.crear_grafo)
        for nombre, real in reales.items():
            assert firma.parameters[nombre].default is real

    async def test_el_supervisor_inyectado_es_el_que_corre(self):
        # Sin mutar nada del módulo: lo que decide es lo que se pasó por parámetro.
        llamadas = []

        def supervisor_espia(state):
            llamadas.append(state["consulta"])
            return {"siguiente": "FINALIZAR", "completado": True,
                    "vueltas": state.get("vueltas", 0) + 1, "messages": []}

        from langgraph.checkpoint.memory import InMemorySaver

        grafo = construccion.crear_grafo(
            supervisor=supervisor_espia, checkpointer=InMemorySaver())
        await grafo.ainvoke(estado_inicial("una consulta de prueba"), config("inyectado"))
        assert llamadas == ["una consulta de prueba"]


class TestRuteo:
    """Las dos aristas condicionales."""

    def test_un_destino_invalido_falla_en_vez_de_ignorarse(self):
        # Sin el mapeo explícito, LangGraph loguea y termina como si nada.
        with pytest.raises(ErrorDeAgente):
            construccion.enrutar({**estado_inicial(), "siguiente": "archivista"})

    def test_finalizar_con_trabajo_listo_pasa_por_aprobacion(self, estado_terminado):
        estado = {**estado_terminado, "siguiente": "FINALIZAR", "completado": True}
        assert construccion.enrutar(estado) == "aprobacion"

    def test_finalizar_sin_trabajo_listo_termina(self, estado_vacio):
        estado = {**estado_vacio, "siguiente": "FINALIZAR", "completado": True}
        assert construccion.enrutar(estado) == "__end__"

    def test_un_trabajo_publicado_cierra(self, estado_terminado):
        assert construccion.enrutar_aprobacion({**estado_terminado, "publicado": True}) == "__end__"

    def test_un_trabajo_sin_publicar_vuelve_al_supervisor(self, estado_terminado):
        assert construccion.enrutar_aprobacion(
            {**estado_terminado, "publicado": False}) == "supervisor"

    def test_falta_publicar_solo_con_el_trabajo_listo(self, estado_terminado, estado_vacio):
        assert construccion.falta_publicar(estado_terminado) is True
        assert construccion.falta_publicar(estado_vacio) is False
        assert construccion.falta_publicar({**estado_terminado, "publicado": True}) is False

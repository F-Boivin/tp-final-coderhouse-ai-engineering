"""El pedido que el investigador arma para el modelo.

El nodo decide qué contexto entra en el prompt, y esa decisión es la que distingue "corregí
las citas que no existen" de "buscá más doctrina". Son instrucciones distintas y no pueden
llegar juntas: el investigador terminaría ampliando cuando lo único pendiente es corregir.

El modelo se reemplaza por un doble que devuelve una investigación fija y guarda el pedido
recibido, así lo que se comprueba es el prompt y no la respuesta.
"""

import pytest

from app.agentes import research_agent
from app.grafo.estado import (
    Aprobacion,
    Cita,
    Investigacion,
    Redaccion,
    Verificacion,
    huella_material,
    huella_texto,
)

CONSULTA = "¿Que es el exceso ritual manifiesto?"
CITAS = (Cita(fallo="Fallos: 311:2437", subseccion="6.2.7 Exceso ritual",
              afirmacion="El fallo sostiene la doctrina aplicable al caso."),)


@pytest.fixture
def pedidos(monkeypatch):
    """Registra el pedido que el nodo le manda al agente, sin llamar a ningún modelo."""
    recibidos = []

    class AgenteFalso:
        async def ainvoke(self, entrada, _config=None):
            recibidos.append(entrada["messages"][0][1])
            return {"structured_response": Investigacion(
                sintesis="Una sintesis de prueba, con largo suficiente.",
                citas=CITAS, subsecciones=("6.2.7 Exceso ritual",)), "messages": []}

    monkeypatch.setattr(research_agent, "construir_investigador", lambda: AgenteFalso())
    return recibidos


def estado_base(**cambios) -> dict:
    base = {"messages": [], "consulta": CONSULTA, "siguiente": "investigador",
            "investigaciones": (), "verificaciones": (), "redacciones": (),
            "aprobaciones": (), "intentos": 0, "vueltas": 0,
            "completado": False, "publicado": False}
    base.update(cambios)
    return base


def con_ampliacion_cumplida() -> dict:
    """El revisor pidió ampliar, el investigador amplió, y el verificador rechazó lo nuevo."""
    inv1 = Investigacion(sintesis="La primera investigacion del trabajo.", citas=CITAS,
                         subsecciones=("6.2.7 Exceso ritual",))
    ver1 = Verificacion(verificadas=("Fallos: 311:2437",), inexistentes=(),
                        aprobado=True, observaciones=())
    red1 = Redaccion(texto="Respuesta escasa.", citas_usadas=("Fallos: 311:2437",),
                     citas_intrusas=(), motivos=(), limpia=True, llamadas=1,
                     sobre_material=huella_material(inv1, ver1))
    humana = Aprobacion(accion="ampliar", revisor="felipe", motivos=("poca doctrina",),
                        sobre_texto=huella_texto(red1.texto),
                        decidida_en="2026-09-02T00:00:00+00:00")
    inv2 = Investigacion(sintesis="La segunda, ya ampliada por el pedido humano.",
                         citas=CITAS, subsecciones=("6.2.7 Exceso ritual",))
    ver2 = Verificacion(verificadas=(), inexistentes=("Fallos: 999:9999",),
                        aprobado=False, observaciones=("la cita no existe en el corpus",))
    return estado_base(investigaciones=(inv1, inv2), verificaciones=(ver1, ver2),
                       redacciones=(red1,), aprobaciones=(humana,), intentos=1, vueltas=5)


def con_ampliacion_pendiente() -> dict:
    """El revisor acaba de pedir ampliar y el investigador todavía no trabajó."""
    inv1 = Investigacion(sintesis="La primera investigacion del trabajo.", citas=CITAS,
                         subsecciones=("6.2.7 Exceso ritual",))
    ver1 = Verificacion(verificadas=("Fallos: 311:2437",), inexistentes=(),
                        aprobado=True, observaciones=())
    red1 = Redaccion(texto="Respuesta escasa.", citas_usadas=("Fallos: 311:2437",),
                     citas_intrusas=(), motivos=(), limpia=True, llamadas=1,
                     sobre_material=huella_material(inv1, ver1))
    humana = Aprobacion(accion="ampliar", revisor="felipe", motivos=("poca doctrina",),
                        sobre_texto=huella_texto(red1.texto),
                        decidida_en="2026-09-02T00:00:00+00:00")
    return estado_base(investigaciones=(inv1,), verificaciones=(ver1,),
                       redacciones=(red1,), aprobaciones=(humana,), vueltas=4)


class TestPedidoDelInvestigador:
    """Qué contexto recibe el modelo, según lo que quedó pendiente."""

    async def test_un_trabajo_nuevo_solo_lleva_la_consulta(self, pedidos):
        await research_agent.investigador_node(estado_base())
        assert pedidos[0] == f"Consulta a investigar: {CONSULTA}"

    async def test_un_rechazo_del_verificador_llega_con_sus_observaciones(self, pedidos):
        rechazo = Verificacion(verificadas=(), inexistentes=("Fallos: 999:9999",),
                               aprobado=False, observaciones=("la cita no existe",))
        inv = Investigacion(sintesis="Una investigacion previa cualquiera.", citas=CITAS,
                            subsecciones=())
        await research_agent.investigador_node(
            estado_base(investigaciones=(inv,), verificaciones=(rechazo,)))
        assert "rechazada por el verificador" in pedidos[0]
        assert "999:9999" in pedidos[0]

    async def test_un_pedido_humano_vigente_llega_como_ampliacion(self, pedidos):
        await research_agent.investigador_node(con_ampliacion_pendiente())
        assert "MÁS DOCTRINA" in pedidos[0]
        assert "poca doctrina" in pedidos[0]

    async def test_un_pedido_de_ampliar_ya_cumplido_no_vuelve_a_llegar(self, pedidos):
        # El revisor pidió ampliar, el investigador amplió y el verificador rechazó lo nuevo:
        # lo único pendiente es corregir. Sumarle "buscá más doctrina" lo mandaría a hacer
        # algo que ya hizo, y contradiría lo que `leer_situacion` le dice al supervisor.
        await research_agent.investigador_node(con_ampliacion_cumplida())
        assert "rechazada por el verificador" in pedidos[0]
        assert "MÁS DOCTRINA" not in pedidos[0]

    async def test_las_dos_instrucciones_nunca_llegan_juntas(self, pedidos):
        for estado in (con_ampliacion_pendiente(), con_ampliacion_cumplida()):
            await research_agent.investigador_node(estado)
        for pedido in pedidos:
            assert not ("rechazada por el verificador" in pedido and "MÁS DOCTRINA" in pedido)


class TestArtefacto:
    """Lo que el nodo deja en el estado."""

    async def test_devuelve_la_investigacion_y_su_linea_de_traza(self, pedidos):
        salida = await research_agent.investigador_node(estado_base())
        assert len(salida["investigaciones"]) == 1
        assert "[investigador]" in salida["messages"][0][1]

    async def test_sin_artefacto_estructurado_falla_como_error_del_agente(self, monkeypatch):
        from app.nucleo.errores import ErrorDeAgente

        class SinArtefacto:
            async def ainvoke(self, *_a, **_k):
                return {"structured_response": None, "messages": []}

        monkeypatch.setattr(research_agent, "construir_investigador", lambda: SinArtefacto())
        with pytest.raises(ErrorDeAgente):
            await research_agent.investigador_node(estado_base())

    async def test_una_caida_del_modelo_se_traduce(self, monkeypatch):
        from app.nucleo.errores import ErrorDeAgente

        class Explota:
            async def ainvoke(self, *_a, **_k):
                raise ConnectionError("sin red")

        monkeypatch.setattr(research_agent, "construir_investigador", lambda: Explota())
        with pytest.raises(ErrorDeAgente):
            await research_agent.investigador_node(estado_base())

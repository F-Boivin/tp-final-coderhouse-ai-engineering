"""Dobles deterministas de los tres especialistas, del supervisor y de Redis.

Los cuatro nodos falsos producen artefactos válidos sin llamar a ningún modelo, así el grafo
real —ruteo, frenos, checkpointer y nodo de publicación— se ejerce en milisegundos. Los usan
la suite de pytest y `verificar_hitl.py`, que es la misma demostración corriendo contra el
sistema entero.

`RedisFalso` implementa los verbos que el proyecto usa. El checkpointer real necesita
RediSearch y RedisJSON, que un doble no reproduce: eso se prueba contra el servicio vivo, en
`demo.py` y en `GET /tasks/{job_id}/checkpoint`.
"""

from datetime import datetime, timezone
from typing import Optional

from redis.exceptions import ConnectionError as RedisConnectionError

from app.grafo.estado import (
    Cita,
    Investigacion,
    Redaccion,
    Verificacion,
    huella_material,
    leer_situacion,
)
from app.nucleo import constantes as cfg

CONSULTA = "¿Que es el exceso ritual manifiesto?"
FALLOS = ("Fallos: 311:2437", "Fallos: 315:1848", "Fallos: 330:1228",
          "Fallos: 338:623", "Fallos: 349:306")
SUBSECCION = "6.2.7 Exceso ritual manifiesto"

# Cuántas veces corrió cada tramo del nodo de publicación. Es lo que prueba la idempotencia.
pasadas = {"antes": 0, "despues": 0}


def estado_inicial(consulta: str = CONSULTA) -> dict:
    """El estado con el que arranca el grafo."""
    return {"messages": [], "consulta": consulta, "siguiente": "investigador",
            "investigaciones": (), "verificaciones": (), "redacciones": (),
            "aprobaciones": (), "intentos": 0, "vueltas": 0,
            "completado": False, "publicado": False}


def citas(cantidad: int) -> tuple:
    """`cantidad` citas del padrón de prueba."""
    return tuple(
        Cita(fallo=f, subseccion=SUBSECCION, afirmacion=f"El fallo {f} sostiene la doctrina.")
        for f in FALLOS[:cantidad]
    )


def investigador_falso(state):
    """Devuelve más citas en cada pasada: es lo que hace observable una ampliación."""
    cantidad = 3 + len(state.get("investigaciones") or ())
    return {"investigaciones": (Investigacion(
        sintesis="Sintesis de prueba sobre el exceso ritual manifiesto.",
        citas=citas(cantidad), subsecciones=(SUBSECCION,)),),
        "messages": [("assistant", f"[investigador] doble · {cantidad} citas")]}


def verificador_falso(state):
    """Aprueba todas las citas de la investigación vigente."""
    investigacion = state["investigaciones"][-1]
    return {"verificaciones": (Verificacion(
        verificadas=tuple(c.fallo for c in investigacion.citas), inexistentes=(),
        aprobado=True, observaciones=()),),
        "messages": [("assistant", "[verificador] doble")]}


def redactor_falso(state):
    """Escribe un texto que invoca todas las citas verificadas."""
    investigacion = state["investigaciones"][-1]
    verificacion = state["verificaciones"][-1]
    n = len(state.get("redacciones") or ()) + 1
    usadas = verificacion.verificadas
    return {"redacciones": (Redaccion(
        texto=f"Redaccion {n} sobre {len(usadas)} citas: {', '.join(usadas)}.",
        citas_usadas=usadas, citas_intrusas=(), motivos=(), limpia=True, llamadas=1,
        sobre_material=huella_material(investigacion, verificacion)),),
        "messages": [("assistant", f"[redactor] doble · {len(usadas)} citas")]}


def redactor_escaso(state):
    """Invoca dos citas mientras haya una sola investigación: por debajo de la holgura.

    Con material ampliado usa todo lo verificado, así se ve que ampliar resolvió el problema.
    El texto lleva el número de versión porque lo que invalida un rechazo es que la huella
    cambie: un redactor que devolviera el mismo texto dejaría el veredicto pendiente.
    """
    salida = redactor_falso(state)
    vieja = salida["redacciones"][0]
    if len(state.get("investigaciones") or ()) > 1:
        return salida
    n = len(state.get("redacciones") or ()) + 1
    usadas = vieja.citas_usadas[:2]
    return {"redacciones": (vieja.model_copy(update={
        "citas_usadas": usadas,
        "texto": f"Redaccion escasa {n} sobre {len(usadas)} citas: {', '.join(usadas)}.",
    }),), "messages": salida["messages"]}


def redactor_sucio(state):
    """Escribe una redacción que la guarda automática rechaza: nunca queda limpia."""
    investigacion = state["investigaciones"][-1]
    verificacion = state["verificaciones"][-1]
    return {"redacciones": (Redaccion(
        texto="Texto con una cita intrusa.", citas_usadas=(), citas_intrusas=("999:9999",),
        motivos=("invoco una cita sin verificar",), limpia=False, llamadas=1,
        sobre_material=huella_material(investigacion, verificacion)),),
        "messages": [("assistant", "[redactor] sucio")]}


def supervisor_falso(state):
    """Delega en orden y cierra cuando hay una redacción vigente y limpia."""
    situacion = leer_situacion(state)
    vueltas = state.get("vueltas", 0) + 1
    intentos = state.get("intentos", 0)
    if intentos >= cfg.MAXIMO_INTENTOS and (
        situacion.rechazo_pendiente or situacion.reescritura_pendiente
    ):
        return {"siguiente": "FINALIZAR", "completado": True, "intentos": intentos,
                "vueltas": vueltas,
                "messages": [("assistant", "[supervisor] tope de intentos")]}
    if situacion.investigacion is None or situacion.ampliacion_pendiente:
        siguiente = "investigador"
        intentos += 1 if situacion.ampliacion_pendiente else 0
    elif not situacion.vigente_verificada:
        siguiente = "verificador"
    elif (situacion.redaccion is None or not situacion.redaccion_vigente
          or situacion.reescritura_pendiente):
        siguiente = "redactor"
        intentos += 1 if situacion.reescritura_pendiente else 0
    else:
        siguiente = "FINALIZAR"
    return {"siguiente": siguiente, "completado": siguiente == "FINALIZAR",
            "intentos": intentos, "vueltas": vueltas,
            "messages": [("assistant", f"[supervisor] -> {siguiente}")]}


def supervisor_con_freno(state):
    """Cierra apenas se agotan los intentos, como el freno real."""
    salida = supervisor_falso(state)
    if salida["intentos"] >= cfg.MAXIMO_INTENTOS:
        salida.update({"siguiente": "FINALIZAR", "completado": True})
    return salida


def aprobacion_contada(nodo_real):
    """Envuelve el nodo real para contar cuántas veces corre cada lado del interrupt."""
    async def contado(state):
        pasadas["antes"] += 1
        salida = await nodo_real(state)
        pasadas["despues"] += 1
        return salida
    return contado


def armar(redactor=redactor_falso, supervisor=supervisor_falso, contar=False):
    """Compila el grafo real con los dobles y un checkpointer en memoria.

    Los cinco nodos entran por parámetro: lo que se prueba es el cableado, el ruteo y el nodo
    de publicación, que son los que no se reemplazan.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    from app.grafo.construccion import crear_grafo
    from app.grafo.hitl import aprobacion_node

    return crear_grafo(
        investigador=investigador_falso,
        verificador=verificador_falso,
        redactor=redactor,
        supervisor=supervisor,
        aprobacion=aprobacion_contada(aprobacion_node) if contar else aprobacion_node,
        checkpointer=InMemorySaver(),
    )


def veredicto(accion: str, motivos=()) -> dict:
    """Lo que manda el revisor al reanudar."""
    return {"accion": accion, "revisor": "felipe", "motivos": list(motivos),
            "decidida_en": datetime.now(timezone.utc).isoformat()}


class RedisFalso:
    """Los verbos de Redis que el proyecto usa, sobre diccionarios en memoria.

    Registra el orden de las operaciones en `traza`, que es lo que permite comprobar que el
    estado de un trabajo se escribe antes de encolarlo.
    """

    def __init__(self) -> None:
        self.claves: dict[str, str] = {}
        self.listas: dict[str, list[str]] = {}
        self.traza: list[tuple] = []
        self.caido = False

    def _control(self) -> None:
        # RedisError y no una excepcion cualquiera: es lo que el proyecto atrapa para
        # traducir a ErrorDeAlmacenamiento y a 503.
        if self.caido:
            raise RedisConnectionError("redis falso caido a proposito")

    async def ping(self) -> bool:
        self._control()
        return True

    async def get(self, clave: str) -> Optional[str]:
        self._control()
        return self.claves.get(clave)

    async def set(self, clave: str, valor: str, nx: bool = False, **_) -> Optional[bool]:
        self._control()
        if nx and clave in self.claves:
            return None
        self.claves[clave] = valor
        self.traza.append(("set", clave))
        return True

    async def rpush(self, lista: str, *valores: str) -> int:
        self._control()
        self.listas.setdefault(lista, []).extend(valores)
        self.traza.append(("rpush", lista, *valores))
        return len(self.listas[lista])

    async def llen(self, lista: str) -> int:
        self._control()
        return len(self.listas.get(lista, []))

    async def lrange(self, lista: str, inicio: int, fin: int) -> list[str]:
        self._control()
        elementos = self.listas.get(lista, [])
        return elementos[inicio:] if fin == -1 else elementos[inicio:fin + 1]

    async def blpop(self, lista: str, timeout: int = 0):
        self._control()
        elementos = self.listas.get(lista, [])
        if not elementos:
            return None
        return (lista, elementos.pop(0))

    async def blmove(self, origen: str, destino: str, timeout: int = 0,
                     src: str = "LEFT", dest: str = "RIGHT") -> Optional[str]:
        self._control()
        elementos = self.listas.get(origen, [])
        if not elementos:
            return None
        valor = elementos.pop(0) if src == "LEFT" else elementos.pop()
        cola = self.listas.setdefault(destino, [])
        cola.append(valor) if dest == "RIGHT" else cola.insert(0, valor)
        self.traza.append(("blmove", origen, destino, valor))
        return valor

    async def lmove(self, origen: str, destino: str,
                    src: str = "LEFT", dest: str = "RIGHT") -> Optional[str]:
        self._control()
        return await self.blmove(origen, destino, 0, src, dest)

    async def lrem(self, lista: str, count: int, valor: str) -> int:
        self._control()
        elementos = self.listas.get(lista, [])
        if valor not in elementos:
            return 0
        elementos.remove(valor)
        self.traza.append(("lrem", lista, valor))
        return 1

    async def scan_iter(self, match: str = "*", **_):
        self._control()
        prefijo = match.rstrip("*")
        for clave in list(self.claves):
            if clave.startswith(prefijo):
                yield clave

    async def aclose(self) -> None:
        return None

"""Esquema del estado compartido del orquestador.

La consigna pide un estado que "permita rastrear qué agente ha contribuido con qué
información". Por eso cada especialista agrega su artefacto a una tupla y las versiones
sucesivas quedan a la vista. Cuando el verificador rechaza y el investigador vuelve a
trabajar, el estado conserva las dos investigaciones, y eso es lo que hace auditable el
ciclo de refinamiento.
"""

import hashlib
import json
from typing import Annotated, Literal, NamedTuple, Optional, Tuple

from langgraph.graph import MessagesState
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Los cuatro destinos que puede elegir el supervisor. Se declara una sola vez: el prompt,
# el tipo de la decisión y el mapeo de las aristas condicionales salen todos de acá.
INVESTIGADOR = "investigador"
VERIFICADOR = "verificador"
REDACTOR = "redactor"
FINALIZAR = "FINALIZAR"
DESTINOS = (INVESTIGADOR, VERIFICADOR, REDACTOR, FINALIZAR)

# El tipo y la tupla se declaran juntos y se controla que no se separen: el Literal no
# puede construirse en runtime desde la tupla, asi que la unica garantia posible es que
# el desajuste falle al importar el modulo y no tres nodos mas adelante.
Destino = Literal["investigador", "verificador", "redactor", "FINALIZAR"]
assert set(Destino.__args__) == set(DESTINOS), "DESTINOS y Destino se desincronizaron"

# El router. Va aparte de DESTINOS porque nadie lo elige como destino: es el nodo al que
# vuelven todos, y el unico que decide.
SUPERVISOR = "supervisor"

# Los tres destinos que son nodos del grafo; FINALIZAR se traduce a END. Se
# controla contra DESTINOS por la misma razon que el Literal: el desajuste tiene que doler
# al importar y no cuando el ruteo no encuentre una clave.
NODOS = (INVESTIGADOR, VERIFICADOR, REDACTOR)
assert set(NODOS) == set(DESTINOS) - {FINALIZAR}, "NODOS y DESTINOS se desincronizaron"

# El nodo que publica. Queda fuera de DESTINOS y de NODOS: el supervisor no lo elige, lo
# impone la arista de cierre. Una pausa que un LLM puede decidir no pedir deja de ser
# obligatoria.
APROBACION = "aprobacion"

# Las tres decisiones que puede tomar el revisor humano.
PUBLICAR = "publicar"
AMPLIAR = "ampliar"
RECHAZAR = "rechazar"
ACCIONES = (PUBLICAR, AMPLIAR, RECHAZAR)


class Cita(BaseModel):
    """Una afirmación del investigador respaldada por un fallo concreto.

    Es el objeto que el verificador contrasta contra los metadatos del corpus. Separar la
    cita de la síntesis es lo que permite comprobarlas una por una, en vez de leer un párrafo
    y confiar.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fallo: str = Field(min_length=3, max_length=120, description='Ej: "Fallos: 311:2437".')
    subseccion: str = Field(min_length=3, max_length=200, description="Subsección del cuadernillo de donde sale.")
    afirmacion: str = Field(min_length=10, max_length=400, description="Qué se sostiene con ese fallo.")


class Investigacion(BaseModel):
    """Lo que el agente de investigación deja en el estado."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sintesis: str = Field(min_length=20, description="Respuesta a la consulta, con la doctrina encontrada.")
    citas: Tuple[Cita, ...] = Field(default_factory=tuple, description="Fallos invocados, uno por afirmación.")
    subsecciones: Tuple[str, ...] = Field(default_factory=tuple, description="Subsecciones consultadas.")

    @field_validator("sintesis")
    @classmethod
    def sin_relleno(cls, valor: str) -> str:
        """Una síntesis en blanco pasa el largo mínimo si viene con espacios."""
        if not valor.strip():
            raise ValueError("la sintesis esta vacia")
        return valor


class Verificacion(BaseModel):
    """El veredicto del verificador sobre las citas de una investigación.

    `aprobado` sale de comparar cada cita contra los metadatos del corpus. Un fallo que no
    está en los metadatos no existe para el sistema, por convincente que suene.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    verificadas: Tuple[str, ...] = Field(default_factory=tuple, description="Fallos que existen en el corpus.")
    inexistentes: Tuple[str, ...] = Field(default_factory=tuple, description="Fallos que el investigador inventó o citó mal.")
    aprobado: bool
    observaciones: Tuple[str, ...] = Field(default_factory=tuple)

    def model_post_init(self, _context) -> None:
        if not self.aprobado and not self.observaciones:
            raise ValueError("un rechazo tiene que venir con al menos una observacion")


class Redaccion(BaseModel):
    """La respuesta final para el usuario, escrita por el redactor.

    Es la fase de síntesis que pide la consigna: el investigador produce material con sus
    citas, el verificador dictamina cuáles resisten el contraste, y recién entonces alguien
    escribe la respuesta. Separar la síntesis de la búsqueda es lo que permite que el texto
    final se apoye **solo en lo verificado**, y no en todo lo que el investigador dijo.

    `limpia` sale de comparar las citas que aparecen en el texto contra las que el
    verificador aprobó. Un redactor que agrega un fallo nuevo en el último paso rompería la
    promesa del sistema justo después de haberla comprobado. Y estar fundado es citar, así
    que la cuenta de las citas que el texto sí usó también entra en el veredicto.

    `llamadas` viaja en el artefacto y no como argumento suelto porque la línea de la traza se
    arma desde acá: si el conteo llegara por otro lado, la demostración del ciclo de corrección
    tendría que reconstruirlo y podría contar otra cosa.

    `sobre_material` es la huella del material del que se escribió, y es lo que permite saber
    si sigue vigente. Es una huella porque lo que invalida una redacción es que **cambie el
    material**: volver a verificar la misma investigación da el mismo veredicto y la deja en
    pie, mientras que una investigación nueva la invalida aunque todavía no se haya
    verificado.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # El piso es 1 y no un número mayor a propósito: "el material verificado no permite
    # responder" es una respuesta legítima y corta, y rechazarla por longitud convertiría una
    # respuesta honesta en una caída del proceso.
    texto: str = Field(min_length=1, description="La respuesta final, en prosa.")
    citas_usadas: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Citas verificadas que el texto efectivamente invoca.",
    )
    citas_intrusas: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Citas que aparecen en el texto y el verificador no aprobó.",
    )
    motivos: Tuple[str, ...] = Field(
        default_factory=tuple, description="Por qué se rechazó, si se rechazó."
    )
    limpia: bool = Field(description="True si el texto está fundado solo en citas verificadas.")
    llamadas: int = Field(
        ge=0, description="Llamadas a herramienta que el redactor hizo para producirlo."
    )
    sobre_material: str = Field(
        min_length=8, description="Huella del material del que se redactó."
    )

    def model_post_init(self, _context) -> None:
        if self.limpia and (self.citas_intrusas or self.motivos):
            raise ValueError("una redaccion limpia no puede tener intrusas ni motivos de rechazo")
        if not self.limpia and not self.motivos:
            raise ValueError("un rechazo tiene que venir con al menos un motivo")


def huella_material(investigacion: "Investigacion", verificacion: "Verificacion") -> str:
    """Identifica el material del que se redacta: la síntesis y las citas aprobadas.

    Si los dos son los mismos, la redacción escrita a partir de ellos sigue valiendo. Si
    cambió cualquiera de los dos, no.
    """
    crudo = json.dumps(
        [investigacion.sintesis, sorted(verificacion.verificadas)], ensure_ascii=False
    )
    return hashlib.sha256(crudo.encode("utf-8")).hexdigest()[:16]


def huella_texto(texto: str) -> str:
    """Identifica el texto exacto que se sometio a aprobacion.

    Es lo que impide que una aprobacion vieja se lea como si aprobara un texto reescrito.
    Mismo mecanismo que `huella_material`, un nivel mas abajo: aca importa el texto final,
    no el material del que salio.
    """
    return hashlib.sha256((texto or "").encode("utf-8")).hexdigest()[:16]


class Aprobacion(BaseModel):
    """El veredicto de un humano sobre un texto concreto.

    Tres acciones. `publicar` cierra el trabajo; `ampliar` lo devuelve al investigador a
    buscar más doctrina; `rechazar` lo devuelve al redactor a reescribir. Las dos últimas
    exigen motivos: sin decir qué falta, "buscá más" es una instrucción vacía.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    accion: Literal["publicar", "ampliar", "rechazar"]
    revisor: str = Field(min_length=1, max_length=80, description="Quien decidio.")
    motivos: Tuple[str, ...] = Field(default_factory=tuple, description="Que hay que corregir.")
    sobre_texto: str = Field(min_length=8, description="Huella del texto que se juzgo.")
    decidida_en: str = Field(min_length=1, description="Marca de tiempo ISO-8601 UTC.")

    @property
    def aprobado(self) -> bool:
        """Si el veredicto autoriza publicar."""
        return self.accion == PUBLICAR

    def model_post_init(self, _context) -> None:
        if self.accion != PUBLICAR and not self.motivos:
            raise ValueError(f"«{self.accion}» tiene que venir con al menos un motivo")


def acumular(actual, nuevo):
    """Reducer que acumula: cada aporte nuevo se agrega al final de los anteriores.

    Es lo que permite mostrar en la traza que hubo un rechazo y qué cambió después. Con el
    default de LangGraph, la segunda investigación borraría a la primera y el ciclo de
    refinamiento sería invisible: se vería el resultado, y no el recorrido.

    Todo se normaliza a tupla antes de concatenar. Hace falta desde que hay checkpointer: el
    estado que vuelve de una pausa pasó por serialización y las tuplas regresan como listas,
    así que sin esta conversión el reducer revienta al reanudar.
    """
    if nuevo is None:
        return tuple(actual or ())
    anteriores = tuple(actual or ())
    agregados = tuple(nuevo) if isinstance(nuevo, (tuple, list)) else (nuevo,)
    return anteriores + agregados


class EstadoOrquestador(MessagesState):
    """Estado global del grafo.

    Hereda de `MessagesState`, así que trae `messages` con el reducer `add_messages` ya
    puesto. Los campos propios:

    - `consulta`: la pregunta del usuario. Entrada, nadie la reescribe.
    - `siguiente`: la decisión del supervisor. Va SIN reducer a propósito: se pisa en cada
      vuelta, que es exactamente lo que se quiere de un campo de ruteo.
    - `investigaciones` / `verificaciones` / `redacciones`: acumulan. Una entrada por
      intento. El `respuesta: str` del código de la clase 6 es acá `redacciones[-1].texto`:
      acumula por la misma razón que las investigaciones — con un campo plano, la segunda
      redacción borraría a la primera y el refinamiento sería invisible.
    - `intentos`: cuántas veces un especialista tuvo que rehacer su trabajo tras un
      rechazo, sumando los dos que pueden recibirlo (el investigador cuando el verificador
      le rechaza las citas, el redactor cuando invoca una cita sin verificar). **Se
      incrementa en un solo lugar: el nodo supervisor, y solo cuando esa vuelta manda a
      rehacer un artefacto que ya fue juzgado.** Sin esa precisión la guarda que lo compara
      no se puede leer.
    - `vueltas`: cuántas veces se ejecutó el supervisor. **Se incrementa en el nodo
      supervisor, en cada ejecución, sin condición.** Es el freno que cubre TODAS las
      ramas: `intentos` solo acota el ciclo de corrección, y un supervisor que delegara
      en círculos sin que nadie rechace nada no lo tocaría nunca.
    - `completado`: el veredicto explícito que lee la arista de cierre. Que el nodo escriba
      su conclusión y la arista solo la lea evita que la condición tenga que re-deducirla.
    - `aprobaciones`: acumula, por lo mismo que las otras tres. Cada veredicto humano queda
      con su motivo y su huella de texto, así el ciclo de aprobación es auditable.
    - `publicado`: el veredicto que lee la arista de salida de la aprobación. Mismo rol que
      `completado`, un nivel más abajo.
    """

    consulta: str
    siguiente: Destino
    investigaciones: Annotated[Tuple[Investigacion, ...], acumular]
    verificaciones: Annotated[Tuple[Verificacion, ...], acumular]
    redacciones: Annotated[Tuple[Redaccion, ...], acumular]
    aprobaciones: Annotated[Tuple[Aprobacion, ...], acumular]
    intentos: int
    vueltas: int
    completado: bool
    publicado: bool


def ultima_investigacion(estado: EstadoOrquestador) -> Optional[Investigacion]:
    """La investigación vigente, o None si el investigador todavía no trabajó."""
    investigaciones = estado.get("investigaciones") or ()
    return investigaciones[-1] if investigaciones else None


def ultima_verificacion(estado: EstadoOrquestador) -> Optional[Verificacion]:
    """El último veredicto del verificador, o None si todavía no revisó."""
    verificaciones = estado.get("verificaciones") or ()
    return verificaciones[-1] if verificaciones else None


def ultima_redaccion(estado: EstadoOrquestador) -> Optional[Redaccion]:
    """La respuesta final vigente, o None si el redactor todavía no escribió."""
    redacciones = estado.get("redacciones") or ()
    return redacciones[-1] if redacciones else None


def ultima_aprobacion(estado: EstadoOrquestador) -> Optional[Aprobacion]:
    """El último veredicto humano, o None si nadie revisó todavía."""
    aprobaciones = estado.get("aprobaciones") or ()
    return aprobaciones[-1] if aprobaciones else None


class Calidad(NamedTuple):
    """Qué tan bien le fue al sistema en este trabajo, en las señales que él mismo produce.

    `critico` es lo que decide si hace falta un humano. Sale de tres umbrales sobre datos que
    ya están en el estado; ninguno estima nada.
    """

    citas_propuestas: int
    citas_verificadas: int
    citas_inexistentes: int
    cobertura: float
    citas_en_el_texto: int
    intentos: int
    motivos: Tuple[str, ...]
    critico: bool


def evaluar_calidad(estado: EstadoOrquestador, holgadas: int,
                    tope_intentos: int) -> Calidad:
    """Mide el trabajo terminado y dice si necesita que lo mire una persona.

    Se calcula solo del estado, sin reloj ni azar: el nodo que la usa se re-ejecuta al
    reanudar y tiene que obtener lo mismo las dos veces.

    **Mide el historial completo, no la última pasada.** Para llegar hasta acá la verificación
    vigente tiene que estar aprobada, y una verificación aprobada no tiene citas inexistentes:
    leer solo la última daría cobertura perfecta en todos los trabajos. Lo que distingue a un
    trabajo difícil de uno fácil es cuántas veces hubo que corregirlo, y eso vive en las
    versiones anteriores que el reducer `acumular` conserva.

    Los tres motivos por los que un trabajo es crítico:

    - **El investigador citó fallos que no existen** en algún momento del recorrido. Las
      corrigió, pero el material lo llevó a inventar.
    - **La respuesta se apoya en pocas citas.** Cumple el mínimo, sin margen.
    - **Se agotaron los intentos de corrección.** El sistema cerró con lo que tenía.

    Un trabajo sin ninguno de los tres se publica sin molestar a nadie.
    """
    verificaciones = estado.get("verificaciones") or ()
    investigacion = ultima_investigacion(estado)
    redaccion = ultima_redaccion(estado)

    verificadas = sum(len(v.verificadas) for v in verificaciones)
    inexistentes = sum(len(v.inexistentes) for v in verificaciones)
    juzgadas = verificadas + inexistentes
    cobertura = verificadas / juzgadas if juzgadas else 0.0
    propuestas = len(investigacion.citas) if investigacion else 0
    en_el_texto = len(redaccion.citas_usadas) if redaccion else 0
    intentos = estado.get("intentos", 0)

    motivos = []
    if inexistentes:
        motivos.append(
            f"el investigador citó {inexistentes} de {juzgadas} fallos que no existen en el "
            f"corpus (cobertura {cobertura:.0%})"
        )
    if en_el_texto < holgadas:
        motivos.append(
            f"la respuesta se apoya en {en_el_texto} citas, por debajo de las {holgadas} "
            f"que dan margen"
        )
    if intentos >= tope_intentos:
        motivos.append(f"se agotaron los {tope_intentos} intentos de corrección")

    return Calidad(
        citas_propuestas=propuestas,
        citas_verificadas=verificadas,
        citas_inexistentes=inexistentes,
        cobertura=cobertura,
        citas_en_el_texto=en_el_texto,
        intentos=intentos,
        motivos=tuple(motivos),
        critico=bool(motivos),
    )


class Situacion(NamedTuple):
    """La lectura del estado que usan el resumen del prompt y los frenos.

    Se calcula una sola vez y la consumen los dos, así que el texto que ve el supervisor y
    la condición que lo frena dicen siempre lo mismo. Cuando cada uno deduce la situación por
    su cuenta, tarde o temprano dejan de coincidir y el sistema cuenta una cosa mientras hace
    otra.
    """

    investigacion: Optional[Investigacion]
    verificacion: Optional[Verificacion]
    redaccion: Optional[Redaccion]
    n_investigaciones: int
    n_verificaciones: int
    vigente_verificada: bool
    redaccion_vigente: bool
    rechazo_pendiente: bool
    rechazo_humano_pendiente: bool
    ampliacion_pendiente: bool
    reescritura_pendiente: bool
    listo: bool


def leer_situacion(state: EstadoOrquestador) -> Situacion:
    """Traduce el estado crudo a las preguntas que el supervisor necesita responder."""
    investigaciones = state.get("investigaciones") or ()
    verificaciones = state.get("verificaciones") or ()
    investigacion = ultima_investigacion(state)
    verificacion = ultima_verificacion(state)
    redaccion = ultima_redaccion(state)

    # El dato que decide el próximo paso es si la investigación VIGENTE ya pasó por el
    # verificador, y no si hubo un rechazo. Sin esta comparación el supervisor no distingue
    # "rechazada y sin corregir" de "rechazada y ya corregida", y vuelve a mandar a
    # investigar sobre una corrección que nunca se revisó.
    vigente_verificada = len(verificaciones) >= len(investigaciones)

    # Y una redacción sigue valiendo mientras el material del que se escribió siga igual. Se
    # compara la huella del material: volver a verificar la misma investigación da el mismo
    # veredicto y la deja en pie, mientras que una investigación nueva la invalida aunque
    # todavía no se haya verificado — caso en que un contador de vueltas ni se habría movido.
    redaccion_vigente = (
        redaccion is not None
        and investigacion is not None
        and verificacion is not None
        and vigente_verificada
        and redaccion.sobre_material == huella_material(investigacion, verificacion)
    )

    # Un veredicto humano solo cuenta contra el texto que se le mostró. Si el redactor ya
    # reescribió, la huella cambió y el veredicto anterior dejó de aplicar — mismo criterio
    # que `redaccion_vigente` sobre el material, un nivel más abajo.
    aprobacion = ultima_aprobacion(state)
    veredicto_vigente = (
        redaccion is not None
        and aprobacion is not None
        and aprobacion.sobre_texto == huella_texto(redaccion.texto)
    )
    # Las dos decisiones que devuelven trabajo se separan porque van a agentes distintos:
    # rechazar es un problema del texto y lo arregla el redactor; ampliar es un problema del
    # material y lo arregla el investigador.
    rechazo_humano_pendiente = veredicto_vigente and aprobacion.accion == RECHAZAR
    # La ampliación además exige que el material siga siendo el mismo. Sin esa cota, la señal
    # sobrevive a la investigación que la resuelve y el supervisor manda a investigar en un
    # bucle: el pedido sigue en pie hasta que el redactor reescriba, que es varios pasos
    # después. El rechazo no la necesita porque `reescritura_pendiente` ya la aplica.
    ampliacion_pendiente = (
        veredicto_vigente and aprobacion.accion == AMPLIAR and redaccion_vigente
    )

    aprobada = verificacion is not None and verificacion.aprobado and vigente_verificada
    return Situacion(
        investigacion=investigacion,
        verificacion=verificacion,
        redaccion=redaccion,
        n_investigaciones=len(investigaciones),
        n_verificaciones=len(verificaciones),
        vigente_verificada=vigente_verificada,
        redaccion_vigente=redaccion_vigente,
        rechazo_pendiente=(
            verificacion is not None and not verificacion.aprobado and vigente_verificada
        ),
        rechazo_humano_pendiente=rechazo_humano_pendiente,
        ampliacion_pendiente=ampliacion_pendiente,
        # Un rechazo humano manda a reescribir igual que uno de la guarda automática: las dos
        # ramas terminan en el redactor, y por eso las dos consumen un intento.
        reescritura_pendiente=(
            aprobada and redaccion_vigente
            and (not redaccion.limpia or rechazo_humano_pendiente)
        ),
        listo=(
            aprobada and redaccion_vigente and redaccion.limpia
            and not rechazo_humano_pendiente and not ampliacion_pendiente
        ),
    )

"""Demostración de la API de producción, golpeándola como lo haría cualquier consumidor.

Habla HTTP contra la API y lee Redis en crudo. No importa ni un nodo del grafo: lo que se
observa es el sistema corriendo en sus procesos, no una simulación dentro de este.

Cada bloque devuelve si cumplió lo que afirma, y el código de salida sale de ese mismo
diccionario: un bloque que falle no puede quedar tapado por el resumen.

Requiere la API y el worker levantados:
    docker compose up --build
    python demo.py
"""

import asyncio
import json
import subprocess
import sys
import time
from typing import Optional

import httpx
import redis.asyncio as redis

import app.nucleo.constantes as cfg
from app.nucleo.config import obtener_ajustes

API = "http://localhost:8000"

# Cinco consultas que el corpus responde con material holgado. Medidas: cada una cierra con
# tres o más citas verificadas y sin correcciones, así que el sistema las publica solo.
CONSULTAS = [
    "¿Que es el exceso ritual manifiesto como causal de arbitrariedad?",
    "¿Cuando una sentencia se aparta del texto legal aplicable?",
    "¿Que exige el requisito de sentencia definitiva?",
    "¿Que es la autocontradiccion como causal de arbitrariedad?",
    "¿Que es el certiorari argentino del articulo 280?",
]
# Dos que el corpus responde con dificultad, y por eso el criterio las manda a revisión.
# La primera es la reproducible: el corpus tiene poca doctrina sobre honorarios de peritos, así
# que la respuesta se apoya en pocas citas corrida tras corrida. La segunda pausa cuando el
# investigador cita algún fallo inexistente, que depende de cómo salga el modelo esa vez.
CONSULTA_FLOJA = "¿Que dijo la Corte sobre los honorarios de peritos?"
CONSULTA_DUDOSA = "¿Que valor tiene la doctrina de la arbitrariedad frente a cuestiones de hecho?"
CONSULTA_MARGINAL = "¿Que dijo la Corte sobre la prescripcion en materia tributaria?"
# Se prueban en orden hasta que alguna se detenga. Cuál lo hace depende de cómo le salga el
# trabajo al sistema esa vez, y esa es justamente la propiedad que se está demostrando.
CANDIDATAS = (CONSULTA_FLOJA, CONSULTA_MARGINAL, CONSULTA_DUDOSA)

TERMINALES = (cfg.COMPLETADA, cfg.FALLIDA)
PAUSA_O_FIN = (cfg.ESPERANDO_APROBACION,) + TERMINALES


def separador(titulo: str) -> None:
    """Encabezado de bloque."""
    print("\n" + "=" * 88)
    print(titulo)
    print("=" * 88)


def recorte(texto: Optional[str], largo: int = 240) -> str:
    """Un texto largo acortado para que la salida siga siendo legible."""
    limpio = " ".join((texto or "").split())
    return limpio if len(limpio) <= largo else limpio[:largo] + " […]"


async def crear(cliente: httpx.AsyncClient, consulta: str) -> str:
    """Encola una consulta y devuelve su job_id."""
    respuesta = await cliente.post("/tasks", json={"consulta": consulta})
    respuesta.raise_for_status()
    return respuesta.json()["job_id"]


async def esperar(cliente: httpx.AsyncClient, job_id: str, estados, limite: int = 90) -> dict:
    """Sondea el estado del trabajo hasta que llegue a alguno de `estados`.

    Es polling contra `GET /tasks/{job_id}`, que es exactamente lo que hace un consumidor de
    una API asincrónica: la respuesta llegó hace rato, el trabajo sigue.
    """
    for _ in range(limite):
        tarea = (await cliente.get(f"/tasks/{job_id}")).json()
        if tarea["estado"] in estados:
            return tarea
        await asyncio.sleep(2)
    return tarea


async def decidir(cliente: httpx.AsyncClient, job_id: str, accion: str, motivos=()) -> int:
    """Manda el veredicto del revisor y devuelve el código HTTP."""
    cuerpo = {"accion": accion, "revisor": "felipe", "motivos": list(motivos)}
    return (await cliente.post(f"/tasks/{job_id}/approve", json=cuerpo)).status_code


async def conseguir_detenido(cliente: httpx.AsyncClient, candidatas,
                             ya_visto=()) -> tuple[Optional[str], dict, list]:
    """Manda consultas hasta que una se detenga, y devuelve cuál lo hizo.

    Un trabajo que se publica solo es el sistema funcionando, no una falla: por eso se prueba
    con varias en vez de dar por sentado que una consulta concreta va a pedir revisión.
    """
    bitacora = []
    for consulta in candidatas:
        if consulta in ya_visto:
            continue
        job_id = await crear(cliente, consulta)
        tarea = await esperar(cliente, job_id, PAUSA_O_FIN, limite=120)
        bitacora.append((consulta, job_id, tarea["estado"]))
        if tarea["estado"] == cfg.ESPERANDO_APROBACION:
            return job_id, tarea, bitacora
    return None, {}, bitacora


async def bloque_0_redis(crudo: redis.Redis, titulo: str) -> bool:
    """[0] Lo que hay adentro de Redis, leído sin intermediarios."""
    separador(titulo)
    print(f"    PING                     -> {await crudo.ping()}")
    print(f"    LLEN {cfg.COLA_TAREAS:<19}-> {await crudo.llen(cfg.COLA_TAREAS)} trabajos en cola")

    estados = [c async for c in crudo.scan_iter(match=cfg.CLAVE_ESTADO.format(job_id='*'))]
    checkpoints = [c async for c in crudo.scan_iter(match=cfg.PATRON_CHECKPOINT, count=500)]
    decisiones = [c async for c in crudo.scan_iter(match="decision:*")]
    print(f"    claves estado:*          -> {len(estados)}")
    print(f"    claves checkpoint*       -> {len(checkpoints)}")
    print(f"    claves decision:*        -> {len(decisiones)}")

    if estados:
        print(f"\n    GET {estados[0]}")
        print(f"      {recorte(await crudo.get(estados[0]), 300)}")
    if checkpoints:
        print(f"\n    una clave del checkpointer: {checkpoints[0]}")
    if estados or checkpoints:
        print("\n    Los tres usos de Redis conviven: cola de trabajo, estado de los trabajos y")
        print("    checkpoints del grafo. Cada uno con su prefijo y su escritor.")
    else:
        print("\n    Arranca vacío: todo lo que aparezca de acá en adelante lo escribieron la")
        print("    API y el worker durante esta corrida.")
    return await crudo.ping()


async def bloque_1_no_bloquea(cliente: httpx.AsyncClient) -> tuple[bool, str]:
    """[1] El POST vuelve en milisegundos mientras el trabajo recién empieza."""
    separador("[1] LA API NO SE BLOQUEA — responde antes de que el trabajo arranque")
    inicio = time.perf_counter()
    respuesta = await cliente.post("/tasks", json={"consulta": CONSULTA_FLOJA})
    demora = time.perf_counter() - inicio
    tarea = respuesta.json()

    print(f"    POST /tasks              -> {respuesta.status_code} en {demora*1000:.0f} ms")
    print(f"    estado devuelto          -> {tarea['estado']}")
    print(f"    job_id                   -> {tarea['job_id']}")
    print(f"    thread_id                -> {tarea['thread_id']}  (uno por trabajo)")
    print("\n    202 y no 200: la tarea fue aceptada, no resuelta. El trabajo que arranca acá")
    print("    tarda decenas de segundos; el cliente ya tiene su respuesta.")
    return respuesta.status_code == 202 and demora < 1.0, tarea["job_id"]


async def bloque_2_concurrencia(cliente: httpx.AsyncClient) -> bool:
    """[2] Cinco consultas a la vez: la prueba de carga que pide la consigna."""
    separador("[2] CINCO CONSULTAS CONCURRENTES — la prueba de carga")
    inicio = time.perf_counter()
    jobs = await asyncio.gather(*(crear(cliente, c) for c in CONSULTAS))
    print(f"    las 5 aceptadas en {(time.perf_counter() - inicio)*1000:.0f} ms")

    # El dato duro de la concurrencia: cuántos estuvieron «processing» al mismo tiempo. Un
    # worker secuencial nunca pasa de uno, por rápido que sea.
    simultaneos = 0
    arranque = time.perf_counter()
    while time.perf_counter() - arranque < 240:
        tareas = (await cliente.get("/tasks")).json()
        mios = [t for t in tareas if t["job_id"] in jobs]
        simultaneos = max(simultaneos, sum(1 for t in mios if t["estado"] == cfg.PROCESANDO))
        if all(t["estado"] in PAUSA_O_FIN for t in mios):
            break
        await asyncio.sleep(1)
    total = time.perf_counter() - inicio

    finales = {t["job_id"]: t for t in (await cliente.get("/tasks")).json() if t["job_id"] in jobs}
    print(f"    máximo en «{cfg.PROCESANDO}» a la vez -> {simultaneos} "
          f"(el worker corre {cfg.CONSUMIDORES} consumidores)")
    print(f"    las 5 llegaron a destino en {total:.1f} s\n")
    publicadas = 0
    for job in jobs:
        t = finales[job]
        publicadas += bool(t["publicado"])
        print(f"      {job[:8]} · {t['estado']:18} publicado={str(t['publicado']):5} "
              f"{len(t['citas'])} citas")
    print(f"\n    {publicadas} de 5 se publicaron sin intervención humana.")
    print(f"    {simultaneos} trabajos en curso a la vez es lo que separa una cola con")
    print("    consumidores concurrentes de un bucle que atiende de a uno.")
    return simultaneos >= 2 and publicadas >= 1


async def bloque_3_criterio(cliente: httpx.AsyncClient,
                            job_flojo: str) -> tuple[bool, str, Optional[str]]:
    """[3] El sistema decide solo cuáles trabajos necesitan una persona."""
    separador("[3] EL CRITERIO — el sistema identifica qué trabajo necesita un humano")
    print(f"    consulta con material holgado : {recorte(CONSULTAS[0], 60)}")
    job_limpio = await crear(cliente, CONSULTAS[0])
    limpio = await esperar(cliente, job_limpio, PAUSA_O_FIN)
    print(f"      -> {limpio['estado']}  ·  publicado={limpio['publicado']}  ·  "
          f"{len(limpio['citas'])} citas  ·  revisiones={limpio['revisiones']}")

    print("\n    consultas de material escaso, hasta que una pida revisión:")
    flojo = await esperar(cliente, job_flojo, PAUSA_O_FIN)
    print(f"      {recorte(CONSULTA_FLOJA, 56):58} -> {flojo['estado']}")
    detenido = job_flojo if flojo["estado"] == cfg.ESPERANDO_APROBACION else None
    if detenido is None:
        detenido, flojo, bitacora = await conseguir_detenido(
            cliente, CANDIDATAS, ya_visto=(CONSULTA_FLOJA,))
        for consulta, _, estado in bitacora:
            print(f"      {recorte(consulta, 56):58} -> {estado}")
    if detenido is None:
        print("\n      ninguna se detuvo: todas se sostuvieron por sí mismas en esta corrida")
        return False, job_limpio, None

    pedido = flojo["aprobacion"]
    print(f"      motivos           : {pedido['motivos']}")
    print(f"      cobertura de citas: {pedido['cobertura']:.0%}")
    print(f"      citas en el texto : {pedido['citas_en_el_texto']} "
          f"(holgura: {cfg.CITAS_HOLGADAS})")
    print(f"      huella del texto  : {pedido['sobre_texto']}")
    print("\n    Uno se publicó solo y el otro espera. La pausa no es un peaje que pagan")
    print("    todos: la impone el propio trabajo cuando no se sostiene por sí mismo.")
    return ((limpio["publicado"] and flojo["estado"] == cfg.ESPERANDO_APROBACION),
            job_limpio, detenido)


async def bloque_4_checkpoint(cliente: httpx.AsyncClient, job_id: str) -> bool:
    """[4] El checkpoint que escribió el worker, leído por la API durante la pausa."""
    separador("[4] EL CHECKPOINT CRUZA PROCESOS — lo escribe el worker, lo lee la API")
    respuesta = await cliente.get(f"/tasks/{job_id}/checkpoint")
    print(f"    GET /tasks/{job_id[:8]}…/checkpoint -> {respuesta.status_code}")
    if respuesta.status_code != 200:
        print(f"    {respuesta.text[:200]}")
        return False
    datos = respuesta.json()
    for clave in ("thread_id", "checkpoint_id", "guardado_en", "escrituras_pendientes", "pausado"):
        print(f"      {clave:22}: {datos[clave]}")
    print(f"      artefactos            : {json.dumps(datos['artefactos'], ensure_ascii=False)}")
    print(f"      claves checkpoint*    : {datos['claves_en_redis']}")
    print("\n    La API nunca corrió este grafo: no tiene el vectorstore ni inicializa las")
    print("    herramientas. Todo lo de arriba lo leyó del checkpoint que dejó el worker.")
    print("    La marca __interrupt__ es lo que le permite a cualquier worker retomarlo.")
    return (datos["checkpoint_id"] != "" and datos["pausado"]
            and "__interrupt__" in datos["escrituras_pendientes"])


async def bloque_5_ampliar(cliente: httpx.AsyncClient, job_id: str) -> tuple[bool, bool]:
    """[5] Ampliar manda al investigador a buscar más material."""
    separador("[5] AMPLIAR — el revisor manda a buscar más doctrina")
    antes = (await cliente.get(f"/tasks/{job_id}")).json()
    citas_antes = len(antes["aprobacion"]["citas"])
    print(f"    citas verificadas antes : {citas_antes}")
    print(f"    texto sometido          : {recorte(antes['aprobacion']['texto'], 180)}")

    motivo = "Solo cita una linea de doctrina; falta el resto del corpus sobre el tema."
    print(f"\n    POST approve (AMPLIAR) -> "
          f"{await decidir(cliente, job_id, 'ampliar', [motivo])}")
    print(f"      motivo: {motivo}")

    despues = await esperar(cliente, job_id, PAUSA_O_FIN, limite=120)
    print(f"\n    estado -> {despues['estado']}  ·  revisiones: {despues['revisiones']}")
    if despues["estado"] == cfg.ESPERANDO_APROBACION:
        nuevo = despues["aprobacion"]
        print(f"    citas verificadas ahora : {len(nuevo['citas'])}")
        print(f"    texto nuevo             : {recorte(nuevo['texto'], 180)}")
        print(f"    huella distinta         : {nuevo['sobre_texto'] != antes['aprobacion']['sobre_texto']}")
    else:
        print(f"    publicado -> {despues['publicado']}  ·  {len(despues['citas'])} citas")
        print(f"    respuesta -> {recorte(despues['resultado'], 200)}")
    print("\n    Ampliar es un problema del material y lo resuelve el investigador; rechazar")
    print("    es un problema del texto y lo resuelve el redactor. Van a agentes distintos.")
    return despues["revisiones"] >= 1, despues["estado"] == cfg.ESPERANDO_APROBACION


async def artefactos(cliente: httpx.AsyncClient, job_id: str) -> dict:
    """Los conteos que el checkpointer guardó para ese trabajo."""
    return (await cliente.get(f"/tasks/{job_id}/checkpoint")).json()["artefactos"]


async def bloque_6_rechazo(cliente: httpx.AsyncClient,
                           reusable: Optional[str] = None) -> tuple[bool, Optional[str]]:
    """[6] Un rechazo vuelve al redactor y obliga a una aprobación nueva.

    Con el presupuesto de vueltas agotado, el freno del supervisor cierra el trabajo antes
    de reescribir: el bloque informa cuál de los dos caminos tomó esta corrida.
    """
    separador("[6] RECHAZAR — el texto vuelve al redactor mientras queden vueltas")
    # Hace falta un trabajo detenido, y detenerse depende de cómo salga el trabajo. Se prueba
    # con la consulta de material escaso, que pausa de forma reproducible; si esta vez sale
    # holgada, se cae a la otra. Un trabajo que se publica solo es el sistema funcionando, no
    # una falla de la demostración.
    # Si el trabajo del bloque anterior sigue detenido, se lo rechaza a él: ya está en el
    # estado que hace falta y no cuesta otra corrida entera del grafo.
    if reusable:
        job_id = reusable
        primera = (await cliente.get(f"/tasks/{job_id}")).json()
        print(f"    {job_id[:8]} · sigue detenido tras la ampliación, se lo rechaza")
    else:
        job_id, primera, bitacora = await conseguir_detenido(cliente, CANDIDATAS)
        for consulta, jid, estado in bitacora:
            print(f"    {jid[:8]} · {recorte(consulta, 52):54} -> {estado}")
        if job_id is None:
            print("    ninguna consulta se detuvo en esta corrida: no hay qué rechazar")
            return False, None

    pedido = primera["aprobacion"]
    antes = await artefactos(cliente, job_id)
    print(f"    motivos de la pausa : {pedido['motivos']}")
    print(f"    primera versión     · huella {pedido['sobre_texto']}")
    print(f"      {recorte(pedido['texto'], 200)}")

    motivo = "Falta explicar el estándar de la Corte antes de aplicarlo al caso."
    print(f"\n    POST approve (RECHAZAR) -> "
          f"{await decidir(cliente, job_id, 'rechazar', [motivo])}")
    print(f"      motivo: {motivo}")

    segunda = await esperar(cliente, job_id, PAUSA_O_FIN, limite=120)
    despues = await artefactos(cliente, job_id)
    print(f"\n    estado tras el rechazo -> {segunda['estado']}  ·  "
          f"revisiones: {segunda['revisiones']}")
    print(f"    redacciones {antes['redacciones']} -> {despues['redacciones']}  ·  "
          f"vueltas {antes['vueltas']} -> {despues['vueltas']} de {cfg.MAXIMO_VUELTAS}")

    if despues["redacciones"] == antes["redacciones"]:
        # El freno general del supervisor corta antes de llamar al modelo. Con el
        # presupuesto de vueltas ya gastado por el ciclo humano, el veredicto queda
        # registrado y el trabajo cierra sin reescribir. Contarlo como reescritura
        # sería maquillar la salida.
        print("    el redactor no corrió: el tope de vueltas cerró el trabajo antes")
        print(f"    publicado -> {segunda['publicado']}")
        return (segunda["revisiones"] > primera["revisiones"]
                and not segunda["publicado"]), None
    if segunda["estado"] != cfg.ESPERANDO_APROBACION:
        print("    el redactor reescribió y el supervisor cerró sin otra aprobación")
        print(f"    publicado -> {segunda['publicado']}")
        return True, None
    nuevo = segunda["aprobacion"]
    print(f"    segunda versión     · huella {nuevo['sobre_texto']}")
    print(f"      {recorte(nuevo['texto'], 200)}")
    print(f"\n    interrupt_id distinto : {nuevo['interrupt_id'] != pedido['interrupt_id']}")
    print(f"    huella distinta       : {nuevo['sobre_texto'] != pedido['sobre_texto']}")
    print("\n    La huella impide publicar un texto con la aprobación de otro: el veredicto")
    print("    vale contra el texto exacto que se le mostró al revisor.")
    return (nuevo["interrupt_id"] != pedido["interrupt_id"]
            and nuevo["sobre_texto"] != pedido["sobre_texto"]), job_id


async def bloque_7_codigos(cliente: httpx.AsyncClient, crudo: redis.Redis,
                           pausado: Optional[str], cerrado: str) -> bool:
    """[7] Los códigos de error, cada uno provocado por su causa real."""
    separador("[7] LOS CÓDIGOS — cada error con su causa")
    resultados = {}

    r = await cliente.get("/tasks/no-existe-este-trabajo")
    resultados["404 · trabajo inexistente"] = (r.status_code, 404)
    print(f"    GET /tasks/no-existe-este-trabajo            -> {r.status_code}")
    print(f"      {r.json()['detail']}")

    r = await cliente.post(f"/tasks/{cerrado}/approve",
                           json={"accion": "publicar", "revisor": "felipe"})
    resultados["409 · el trabajo no espera aprobación"] = (r.status_code, 409)
    print(f"\n    POST approve sobre un trabajo ya cerrado     -> {r.status_code}")
    print(f"      {r.json()['detail']}")

    r = await cliente.post("/tasks", json={"consulta": "corta"})
    resultados["422 · consulta demasiado corta"] = (r.status_code, 422)
    print(f"\n    POST /tasks con una consulta de 5 caracteres -> {r.status_code}")

    r = await cliente.post(f"/tasks/{cerrado}/approve",
                           json={"accion": "rechazar", "revisor": "felipe", "motivos": []})
    resultados["422 · rechazo sin motivos"] = (r.status_code, 422)
    print(f"    POST approve rechazando sin dar motivos      -> {r.status_code}")
    print("      un rechazo sin motivo mandaría a reescribir sin decir qué corregir")

    r = await cliente.post(f"/tasks/{cerrado}/approve",
                           json={"accion": "opinar", "revisor": "felipe", "motivos": ["x"]})
    resultados["422 · acción inexistente"] = (r.status_code, 422)
    print(f"    POST approve con una acción que no existe    -> {r.status_code}")

    if pausado:
        iid = (await cliente.get(f"/tasks/{pausado}")).json()["aprobacion"]["interrupt_id"]
        a, b = await asyncio.gather(
            cliente.post(f"/tasks/{pausado}/approve",
                         json={"accion": "publicar", "revisor": "ana"}),
            cliente.post(f"/tasks/{pausado}/approve",
                         json={"accion": "publicar", "revisor": "beto"}),
        )
        codigos = sorted([a.status_code, b.status_code])
        resultados["202+409 · dos revisores a la vez"] = (tuple(codigos), (202, 409))
        print(f"\n    dos POST approve CONCURRENTES sobre la misma pausa -> {codigos}")
        perdedor = a if a.status_code == 409 else b
        print(f"      {perdedor.json()['detail']}")
        print("      Uno gana y el otro se entera: la publicación ocurre una sola vez.")

        # Contra una pausa hay dos cerrojos, y el mensaje de arriba delata cuál actuó. El de
        # estado alcanza casi siempre porque reanudar una decisión no llama al modelo: el
        # trabajo ya cerró cuando entra el segundo pedido. El otro cubre la ventana en que los
        # dos entran antes de que el worker toque el estado, y se ve acá: sobre una clave que
        # ya existe, un SET NX no escribe.
        clave = cfg.CLAVE_DECISION.format(job_id=pausado, interrupt_id=iid)
        guardada = await crudo.get(clave)
        reintento = await crudo.set(clave, '{"accion": "rechazar"}', nx=True)
        print(f"\n    GET {clave[:52]}…")
        print(f"      {recorte(guardada, 160)}")
        print(f"    SET … NX sobre esa misma clave -> {reintento}  (no escribe: ya está decidida)")
        resultados["el SET NX no pisa una decisión tomada"] = (reintento, None)

    for nombre, (obtenido, esperado) in resultados.items():
        if obtenido != esperado:
            print(f"    DISTINTO DE LO ESPERADO: {nombre} dio {obtenido}, se esperaba {esperado}")
    return all(o == e for o, e in resultados.values())


async def bloque_8_falla(cliente: httpx.AsyncClient, crudo: redis.Redis) -> bool:
    """[8] Un trabajo que el sistema no puede resolver queda en `failed` con su motivo.

    La falla se provoca con una inconsistencia real y reproducible: un trabajo que dice estar
    esperando una decisión que no existe en Redis. Esperar a que una consulta le salga mal al
    modelo haría que este bloque pasara o fallara según la suerte de la corrida.
    """
    separador("[8] EL CAMINO INFELIZ — un fallo termina el trabajo, no el servicio")
    job_id = "roto-a-proposito"
    tarea = {
        "job_id": job_id, "consulta": "Un trabajo con una pausa sin decision guardada.",
        "estado": cfg.PENDIENTE, "thread_id": job_id,
        "creada_en": "2026-08-28T00:00:00+00:00", "actualizada_en": "2026-08-28T00:00:00+00:00",
        "resultado": None, "citas": [], "publicado": False, "revisiones": 0, "error": None,
        "aprobacion": {
            "interrupt_id": "una-pausa-sin-decision", "consulta": "x", "texto": "y",
            "citas": [], "sobre_texto": "0000000000000000", "intentos": 0,
            "motivos": [], "cobertura": 0.0, "citas_en_el_texto": 0,
            "solicitada_en": "2026-08-28T00:00:00+00:00",
        },
    }
    await crudo.set(cfg.CLAVE_ESTADO.format(job_id=job_id), json.dumps(tarea))
    await crudo.rpush(cfg.COLA_TAREAS, job_id)
    print(f"    encolado {job_id}: dice esperar una decisión que no existe en Redis")
    final = await esperar(cliente, job_id, TERMINALES, limite=40)
    print(f"    estado -> {final['estado']}")
    print(f"    error  -> {recorte(final['error'], 200)}")

    salud = (await cliente.get("/health")).json()
    print(f"\n    /health después del fallo -> {salud['estado']}")
    print("    El worker siguió tomando trabajos. Un fallo termina un trabajo, no el servicio.")
    return final["estado"] == cfg.FALLIDA and bool(final["error"])


async def bloque_9_worker_muerto(cliente: httpx.AsyncClient, crudo: redis.Redis) -> bool:
    """[9] Matar al worker a mitad de un trabajo no lo pierde: vuelve solo al reiniciar.

    Prueba el checkpointer y la cola de proceso a la vez. Con `BLPOP` el id desaparecía de
    Redis al tomarlo, y un worker muerto dejaba el trabajo en `processing` para siempre; con
    `BLMOVE` queda anotado en `cola:procesando`, que es de donde lo recupera el arranque
    siguiente. El trabajo lo termina un proceso distinto del que lo empezó.

    Necesita el CLI de docker, igual que el resto de la demostración necesita el compose.
    """
    separador("[9] UN WORKER MUERTO NO PIERDE EL TRABAJO — checkpoint + cola de proceso")
    job_id = await crear(cliente, CONSULTAS[0])
    print(f"    encolado {job_id}")

    # Esperar a que un consumidor lo tome: ahí es cuando el id está en la cola de proceso.
    en_proceso: list[str] = []
    for _ in range(60):
        en_proceso = await crudo.lrange(cfg.COLA_PROCESANDO, 0, -1)
        if job_id in en_proceso:
            break
        await asyncio.sleep(0.5)
    print(f"    LRANGE {cfg.COLA_PROCESANDO} -> {en_proceso}")
    if job_id not in en_proceso:
        print("    el trabajo nunca apareció en la cola de proceso")
        return False

    matado = subprocess.run(["docker", "compose", "kill", "worker"],
                            capture_output=True, text=True)
    if matado.returncode != 0:
        print(f"    no se pudo matar el worker: {matado.stderr.strip()[:120]}")
        return False
    print("    docker compose kill worker  ·  el proceso murió con el trabajo entre manos")

    await asyncio.sleep(2)
    estado = (await cliente.get(f"/tasks/{job_id}")).json()
    sigue_anotado = job_id in await crudo.lrange(cfg.COLA_PROCESANDO, 0, -1)
    print(f"    estado del trabajo -> {estado['estado']}")
    print(f"    sigue anotado en la cola de proceso -> {sigue_anotado}")

    subprocess.run(["docker", "compose", "start", "worker"], capture_output=True, text=True)
    print("    docker compose start worker  ·  el arranque recupera lo que quedó a medias")

    final = await esperar(cliente, job_id, TERMINALES + (cfg.ESPERANDO_APROBACION,), limite=180)
    pendientes = await crudo.llen(cfg.COLA_PROCESANDO)
    print(f"    estado final -> {final['estado']}")
    print(f"    cola de proceso al terminar -> {pendientes} elementos")
    print("\n    El trabajo lo retomó un proceso distinto del que lo empezó, desde el")
    print("    checkpoint que había quedado en Redis.")
    return sigue_anotado and final["estado"] != cfg.PROCESANDO and pendientes == 0


async def demostrar() -> int:
    """Corre los diez bloques y devuelve el código de salida."""
    print("=" * 88)
    print("API DE PRODUCCIÓN — orquestador multi-agente con revisión humana condicional")
    print("=" * 88)
    crudo = redis.from_url(obtener_ajustes().redis_url, decode_responses=True)
    async with httpx.AsyncClient(base_url=API, timeout=30) as cliente:
        try:
            salud = await cliente.get("/health")
        except httpx.ConnectError:
            print(f"ERROR: {cfg.ERROR_API_APAGADA}", file=sys.stderr)
            return 1
        print(f"\n/health -> {salud.status_code} · {json.dumps(salud.json(), ensure_ascii=False)}")

        resultados = {"[0] Redis por dentro": await bloque_0_redis(
            crudo, "[0] REDIS POR DENTRO — la cola, los estados y los checkpoints")}
        resultados["[1] la API no se bloquea"], flojo = await bloque_1_no_bloquea(cliente)
        resultados["[2] cinco consultas concurrentes"] = await bloque_2_concurrencia(cliente)
        # El trabajo limpio de [3] es el único que queda cerrado a esta altura: los demás
        # siguen esperando decisión, y el 409 de «no espera aprobación» necesita uno cerrado.
        resultados["[3] el criterio identifica el trabajo crítico"], cerrado, detenido = \
            await bloque_3_criterio(cliente, flojo)
        # Los dos que siguen necesitan un trabajo efectivamente detenido, y cuál se detiene
        # depende de la corrida.
        sigue_detenido = False
        if detenido is None:
            print("\n[4] y [5] necesitan un trabajo detenido, y esta corrida no produjo ninguno.")
            resultados["[4] el checkpoint cruza procesos"] = False
            resultados["[5] ampliar vuelve al investigador"] = False
        else:
            resultados["[4] el checkpoint cruza procesos"] = await bloque_4_checkpoint(
                cliente, detenido)
            resultados["[5] ampliar vuelve al investigador"], sigue_detenido = \
                await bloque_5_ampliar(cliente, detenido)
        resultados["[6] el rechazo humano llega al grafo"], pausado = await bloque_6_rechazo(
            cliente, detenido if sigue_detenido else None)
        resultados["[7] los códigos de error"] = await bloque_7_codigos(
            cliente, crudo, pausado, cerrado)
        resultados["[8] el camino infeliz"] = await bloque_8_falla(cliente, crudo)
        resultados["[9] un worker muerto no pierde el trabajo"] = \
            await bloque_9_worker_muerto(cliente, crudo)

        if pausado:
            await esperar(cliente, pausado, TERMINALES, limite=30)
        await bloque_0_redis(crudo, "REDIS AL CERRAR — lo que quedó escrito")

    await crudo.aclose()
    separador("RESUMEN")
    for nombre, cumplio in resultados.items():
        print(f"    {'OK   ' if cumplio else 'FALLA'}  {nombre}")
    if not all(resultados.values()):
        print(f"\n{cfg.ERROR_DEMO_INCOMPLETA}", file=sys.stderr)
        return 1
    print(f"\nLos {len(resultados)} bloques hicieron lo que afirman.")
    return 0


def main() -> int:
    """Punto de entrada."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    return asyncio.run(demostrar())


if __name__ == "__main__":
    sys.exit(main())

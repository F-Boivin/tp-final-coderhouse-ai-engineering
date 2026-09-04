"""Configuracion y mensajes del orquestador. Todo lo ajustable vive aca."""

from pathlib import Path

# La raiz del repositorio: constantes.py vive en app/, y data/ y vectorstore/ estan
# un nivel mas arriba.
RAIZ = Path(__file__).parent.parent.parent

# --- Datos y persistencia ---
DIRECTORIO_DATA = RAIZ / "data"
PATRON_DOCUMENTOS = "*.md"

# Los modelos, el proveedor, la coleccion y las rutas de servicio se configuran por entorno:
# viven en `Ajustes` (config.py). Aca queda lo que es invariante del sistema.

# --- Chunking: los extractos de doctrina son indivisibles ---
TAMANO_CHUNK_TOKENS = 500
SOLAPAMIENTO_CHUNK_TOKENS = 50
# El primer separador es el limite entre extractos de doctrina: el splitter parte ahi
# antes que en cualquier otro lado, asi ningun extracto se corta al medio.
SEPARADOR_EXTRACTOS = "\n\n---\n\n"
SEPARADORES = [SEPARADOR_EXTRACTOS, "\n\n", "\n", " ", ""]
ETIQUETA_FUENTE = "Fuente:"
# Distancia coseno explicita (leccion LCEL 2: la intencion se escribe).
METRICA_DISTANCIA = {"hnsw:space": "cosine"}

# --- Herramientas del agente ---
RESULTADOS_RECUPERADOS = 4
# Candidatos que aporta cada lado del ensamble antes de fusionar. Mas altos que el resultado
# final a proposito: la fusion elige mejor viendo mas de cada uno.
CANDIDATOS_POR_RETRIEVER = 10
PESO_BM25 = 0.5             # los dos lados pesan igual: el lexico encuentra la cita escrita
PESO_VECTORIAL = 0.5        # literal, el vectorial la consulta parafraseada
MAXIMO_RESULTADOS = 8
LARGO_MAXIMO_FRAGMENTO = 900  # caracteres por fragmento en la observacion
LARGO_MAXIMO_CONSULTA = 500

# --- Mensajes de ingesta ---
MENSAJE_SIN_DATASET = (
    "No hay documentos {patron} en {ruta}. El corpus viaja en el repositorio: si la carpeta "
    "está vacía, la copia del repo quedó incompleta."
)
MENSAJE_INDICE_EXISTENTE = (
    "Índice existente en {ruta}: {cantidad} fragmentos ya indexados, no se reindexa."
)
MENSAJE_INDEXANDO = "No hay índice previo en {ruta}: indexando por primera vez."
MENSAJE_DOCUMENTOS_LEIDOS = "Documentos leídos: {cantidad} ({tokens} tokens en total)"
MENSAJE_CHUNKS = (
    "Fragmentos generados: {cantidad} · tokens por fragmento: "
    "mín {minimo}, promedio {promedio}, máx {maximo} (techo configurado: {techo})"
)
MENSAJE_INDICE_LISTO = "Índice listo: {cantidad} fragmentos en la colección '{coleccion}'."

# --- Mensajes de las herramientas ---
ENCABEZADO_SUBSECCIONES = "Subsecciones de estos fragmentos (para fallos_citados):"
MENSAJE_SIN_RESULTADOS = (
    "La búsqueda no devolvió ningún fragmento para esa consulta. "
    "Probá reformularla con otros términos."
)
MENSAJE_SIN_SUBSECCION = (
    "No existe la subsección «{subseccion}» en el índice. Elegí uno de estos y reintentá: "
    "{validas}"
)
ENCABEZADO_FALLOS = "Fallos citados en «{subseccion}» (fuente: metadatos del índice):"

# --- Mensajes de error (granulares, por tipo) ---
ERROR_CLAVE_DE_PROVEEDOR = (
    "Falta {variable}. Copiá .env.example a .env y cargá tu clave "
    "(el .env nunca se commitea)."
)
ERROR_LIMITE = "Límite de uso de la API del proveedor alcanzado: {detalle}"
ERROR_CONEXION = "No se pudo conectar con la API del proveedor: {detalle}"
ERROR_API = "La API del proveedor devolvió un error: {detalle}"
ERROR_VECTORSTORE = "Error de la base vectorial: {detalle}"
ERROR_LECTURA_DATASET = "No se pudo leer {archivo} del dataset: {detalle}"
ERROR_DOCUMENTO_VACIO = (
    "El documento {archivo} de data/ está vacío. Restaurá el corpus desde el repositorio."
)
ERROR_HERRAMIENTA_BASE = (
    "La base de conocimiento no está disponible ({detalle}). "
    "Informale el problema al usuario o intentá de nuevo más tarde."
)
ERROR_HERRAMIENTA_NO_INICIALIZADA = (
    "Las herramientas no fueron inicializadas: llamá a herramientas.inicializar() "
    "con el vectorstore antes de construir el grafo."
)
ERROR_RECUPERADOR_SINCRONICO = (
    "Este recuperador es asincronico: usa ainvoke. El camino sincronico bloquearia el "
    "event loop."
)
ERROR_HERRAMIENTA_SIN_EMBEDDINGS = (
    "La base vectorial no tiene funcion de embeddings: no se puede consultar por similitud."
)


# --- El orquestador: supervisor, especialistas y sus frenos ---
MAXIMO_INTENTOS = 3                 # veces que un especialista puede tener que rehacer su
                                    # trabajo, sumando las dos correcciones posibles (citas
                                    # rechazadas al investigador, cita sin verificar al
                                    # redactor), antes de cerrar con lo que haya
CITAS_MINIMAS = 2                   # por debajo, la respuesta es correcta pero floja
LIMITE_RECURSION_AGENTE = 12        # supersteps del ReAct interno del investigador
LIMITE_RECURSION_REDACTOR = 12      # supersteps del ReAct interno del redactor: una vuelta
                                    # por cita para pedir su link, mas la redaccion final
LIMITE_RECURSION_ORQUESTADOR = 30   # supersteps del grafo del orquestador. La guarda del
                                    # supervisor es `vueltas > MAXIMO_VUELTAS`, asi que la
                                    # vuelta 13 SE EJECUTA y recien ahi cierra. Cada
                                    # vuelta arrastra a lo sumo un superstep detras —un
                                    # especialista, o el nodo de aprobacion—, asi que el peor
                                    # caso es 2*(MAXIMO_VUELTAS+1) = 26. El HITL entra en esa
                                    # cuenta sin ampliarla: la aprobacion ocupa el lugar del
                                    # especialista que esa vuelta no delego, y un rechazo
                                    # humano vuelve al supervisor, que ya esta contado. Es
                                    # una red, no la condicion de corte. Ojo al tocar
                                    # MAXIMO_VUELTAS: desde 14 la formula pasa de 30 y la
                                    # corrida cortaria por GraphRecursionError en vez de
                                    # cerrar ordenada.


MOTIVO_CITAS_INTRUSAS = "invoco citas que el verificador no aprobo: {citas}"
MOTIVO_POCAS_CITAS = (
    "el texto se apoya en {usadas} cita(s) verificada(s); se piden al menos {minimas} para "
    "darlo por fundado"
)
ERROR_SUPERVISOR = "El supervisor no pudo decidir el proximo paso: {detalle}"
ERROR_DECISION_INCOMPLETA = (
    "la respuesta se corto por limite de tokens y la decision puede estar a medias"
)
ERROR_DECISION_VACIA = "el modelo no devolvio ninguna decision"
INTENTOS_DECISION = 3       # llamadas al modelo por decision del supervisor, con backoff
                            # exponencial y jitter entre una y otra
ERROR_INVESTIGADOR = "El investigador fallo: {detalle}"
ERROR_VERIFICADOR = "El verificador no pudo consultar el padron de citas: {detalle}"
ERROR_VERIFICADOR_SIN_INVESTIGACION = (
    "El verificador se ejecuto sin investigacion en el estado: el supervisor lo delego "
    "fuera de orden."
)
ERROR_REDACTOR = "El redactor no pudo escribir la respuesta final: {detalle}"
ERROR_REDACTOR_SIN_MATERIAL = (
    "El redactor se ejecuto sin investigacion o sin verificacion en el estado: el "
    "supervisor lo delego fuera de orden. Redactar antes de verificar seria escribir sobre "
    "citas que todavia nadie comprobo."
)
MENSAJE_SIN_CITAS_APROBADAS = "(ninguna: el verificador no aprobo ninguna cita)"

# Respuestas de link_oficial. Las tres son observaciones para el modelo, no errores: el
# redactor tiene que poder seguir escribiendo sin el link en vez de cortar la corrida.
MENSAJE_CITA_NO_APROBADA = (
    "'{cita}' no esta entre las citas verificadas de esta consulta, asi que no tiene link ni "
    "puede aparecer en la respuesta. Usa solo las de la lista."
)
MENSAJE_CITA_ILEGIBLE = (
    "No pude leer un numero de fallo en '{cita}'. Pedimelo como 'Fallos: tomo:pagina'."
)
MENSAJE_SIN_LINK = (
    "'{cita}' esta verificada pero el corpus no registra su link oficial. Citala igual, sin "
    "link."
)
ERROR_DESTINO_INVALIDO = "El supervisor eligio un destino que no existe: {destino!r}"

ERROR_LIMITE_ORQUESTADOR = (
    "Se alcanzo el limite de supersteps del orquestador sin que el supervisor cerrara. "
    "El limite es una red, no la condicion de corte: si salta, el problema esta en el ruteo."
)

MAXIMO_VUELTAS = 12                 # freno general del supervisor: cubre todas las ramas,
                                    # incluida la aprobada, que MAXIMO_INTENTOS no toca.
                                    # El camino feliz son 4 vueltas (investigador,
                                    # verificador, redactor, cierre) y cada correccion suma
                                    # como mucho 2, asi que con MAXIMO_INTENTOS=3 el peor
                                    # caso legitimo son 10: el tope deja margen y sigue
                                    # cortando un supervisor que delegue en circulos.


# --- Observabilidad (LangSmith) ---
AVISO_LANGSMITH_APAGADO = (
    "AVISO: sin LANGSMITH_API_KEY el sistema corre pero no deja trazas."
)
AVISO_LANGSMITH_INALCANZABLE = (
    "AVISO: LangSmith no responde; el sistema sigue, las trazas se pierden."
)
ERROR_APROBACION_INVALIDA = "La decision de aprobacion no tiene la forma esperada: {detalle}"


ERROR_HITL_INCOMPLETO = (
    "Algun bloque del ciclo de aprobacion no hizo lo que afirma: revisar antes de seguir."
)


# --- La capa de produccion: Redis, la cola y los estados de un trabajo ---
COLA_TAREAS = "cola:tareas"
# Donde queda anotado un trabajo mientras un consumidor lo procesa. Es lo que permite
# devolverlo a la cola si el worker muere sin llegar a un estado terminal.
COLA_PROCESANDO = "cola:procesando"
CLAVE_ESTADO = "estado:{job_id}"
CLAVE_DECISION = "decision:{job_id}:{interrupt_id}"
PATRON_CHECKPOINT = "checkpoint*"

# Los cinco estados de un trabajo. Los valores van en ingles porque son la API publica: es
# lo que lee quien consulta el endpoint.
PENDIENTE = "pending"
PROCESANDO = "processing"
ESPERANDO_APROBACION = "awaiting_approval"
COMPLETADA = "completed"
FALLIDA = "failed"

CONSUMIDORES = 3            # corrutinas del worker en un mismo event loop: todos los nodos
                            # son async, asi que la concurrencia es real sin multiplicar
                            # Chroma, el padron ni el checkpointer
ESPERA_COLA = 5             # segundos que el BLMOVE bloquea antes de volver a mirar el apagado
LARGO_MAXIMO_MOTIVO = 300   # por motivo de rechazo que manda el revisor

ERROR_REDIS = "Redis no esta disponible: {detalle}"
ERROR_TAREA_INEXISTENTE = "No existe ningun trabajo con id {job_id}."
ERROR_TAREA_SIN_APROBACION = (
    "El trabajo {job_id} esta en «{estado}» y no espera ninguna aprobacion."
)
ERROR_APROBACION_YA_DECIDIDA = "La aprobacion del trabajo {job_id} ya fue decidida."
ERROR_ESTADO_ILEGIBLE = "El estado guardado del trabajo {job_id} no se puede leer: {detalle}"
ERROR_CHECKPOINT_AUSENTE = "El trabajo {job_id} todavia no dejo ningun checkpoint."
AVISO_WORKER_ARRANCA = "Worker listo: {consumidores} consumidores sobre {url}"
AVISO_WORKER_APAGA = "Worker apagandose: no se toman trabajos nuevos."
AVISO_HUERFANO_RECUPERADO = (
    "Recuperado el trabajo {job_id}: quedo en proceso de un worker anterior y vuelve a la cola."
)
AVISO_TRAZAS_SIN_VACIAR = "AVISO: quedaron trazas sin enviar a LangSmith ({detalle})."
ERROR_DECISION_AUSENTE = (
    "El trabajo {job_id} volvio a la cola esperando aprobacion, pero no hay ninguna decision "
    "guardada para esa pausa."
)
TITULO_API = "Orquestador REX — API de produccion"
DESCRIPCION_API = (
    "Consultas sobre doctrina del recurso extraordinario federal, resueltas por un equipo "
    "de agentes. La API acepta y consulta; el worker ejecuta. Publicar requiere la "
    "aprobacion de un revisor humano."
)
VERSION_API = "8.0"
ERROR_MOTIVO_LARGO = "Cada motivo tiene un tope de {tope} caracteres."
ERROR_API_APAGADA = (
    "La API no responde en localhost:8000. Levantala antes: docker compose up --build"
)
ERROR_FALLO_NO_CAZADO = (
    "El verificador no cazo la cita inventada: la guarda anti-alucinacion no esta haciendo "
    "su trabajo."
)
ERROR_PROVEEDOR_NO_VERIFICADO = (
    "El factory no construyo el cliente esperado o el grafo no produjo respuesta con el "
    "proveedor alternativo."
)
ERROR_HIBRIDO_INCOMPLETO = (
    "Algun bloque de la verificacion del hibrido no hizo lo que afirma: revisar antes de seguir."
)
ERROR_DEMO_INCOMPLETA = (
    "Algun bloque de la demostracion no hizo lo que afirma: revisar antes de entregar."
)

# --- El criterio que manda un trabajo a revision humana ---
# La consigna pide una pausa humana "en tareas que el sistema identifique como criticas". El
# sistema las identifica midiendo su propio trabajo: un resultado limpio se publica solo, uno
# flojo se detiene. Los tres umbrales estan en `evaluar_calidad` de grafo/estado.py.
CITAS_HOLGADAS = 3      # citas en el texto final por debajo de las cuales se pide revision.
                        # Es CITAS_MINIMAS + 1: cumplir el minimo justo no deja margen, y una
                        # respuesta al limite es la que conviene que mire una persona.

MOTIVO_SIN_REVISION = "publicada sin revision: el trabajo no dio ninguna senal de alerta"
ERROR_ACCION_INVALIDA = (
    "La accion «{accion}» no existe. Las validas son: publicar, ampliar, rechazar."
)
ERROR_DECISION_SIN_MOTIVOS = "«{accion}» tiene que venir con al menos un motivo."
AVISO_ESTADOS_ILEGIBLES = (
    "AVISO: estas claves de estado no se pudieron leer y quedaron fuera del listado: {claves}"
)
MAXIMO_MOTIVOS = 20     # motivos por decision: cada uno ya tiene tope de largo, la lista no

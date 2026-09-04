# Una sola imagen para los dos servicios: api y worker corren el mismo codigo y solo cambia
# el comando. Dos imagenes obligarian a mantener dos listas de dependencias en sincronia.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /srv

# Las dependencias antes que el codigo: mientras requirements.txt no cambie, editar un modulo
# reusa la capa de instalacion en vez de recompilar todo.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/

# El directorio del indice se crea en la imagen aunque llegue vacio. Docker le da a un
# volumen nombrado el dueno y los permisos de la ruta que tapa; si la ruta no existiera, la
# crearia como root y el worker no podria escribir el indice.
RUN mkdir -p /srv/vectorstore

# Sin usuario root: si el contenedor se compromete, el atacante no hereda privilegios.
RUN useradd --create-home orquestador && chown -R orquestador:orquestador /srv
USER orquestador

CMD ["python", "-m", "app.trabajos.worker"]

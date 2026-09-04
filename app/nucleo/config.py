"""La configuración que llega por entorno, validada con Pydantic.

`Ajustes` recoge lo que cambia entre despliegues —proveedor, modelos, claves, Redis, la
colección del índice y el proyecto de trazas— y lo valida al construirse. Lo que no está acá
es invariante del sistema y vive en `constantes.py`.

Se lee una vez por proceso: `obtener_ajustes()` cachea la instancia, y nadie la construye al
importar, así la validación ocurre cuando el proceso arranca.

Verificado contra la documentación oficial (pydantic-settings 2.14.2):
- `BaseSettings` toma los valores del entorno y de `env_file`, con el nombre del campo en
  mayúsculas. https://docs.pydantic.dev/latest/concepts/pydantic_settings/
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

import app.nucleo.constantes as cfg
from app.nucleo.errores import ErrorDeConfiguracion


class Proveedor(str, Enum):
    """Los proveedores de chat que el factory sabe construir."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Ajustes(BaseSettings):
    """Todo lo que se configura desde afuera del código.

    Los modelos por rol admiten `None`: sin valor, cada uno toma el modelo por defecto del
    proveedor activo. Con un default fijo, pedir `LLM_PROVIDER=anthropic` dejaría al sistema
    pidiéndole a Anthropic un modelo de OpenAI.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- Proveedor y claves ---
    llm_provider: Proveedor = Proveedor.OPENAI
    openai_api_key: Optional[SecretStr] = None
    anthropic_api_key: Optional[SecretStr] = None

    # --- Modelos por rol ---
    modelo_supervisor: Optional[str] = None
    modelo_investigador: Optional[str] = None
    modelo_redactor: Optional[str] = None
    modelo_embeddings: str = "text-embedding-3-small"
    temperatura: float = Field(default=0.0, ge=0.0, le=2.0)

    # --- Infraestructura ---
    redis_url: str = "redis://localhost:6379/0"
    nombre_coleccion: str = "csjn_sentencias_arbitrarias"
    directorio_vectorstore: Path = cfg.RAIZ / "vectorstore"

    # --- Observabilidad ---
    langsmith_api_key: Optional[SecretStr] = None
    proyecto_langsmith: str = "orquestador-rex-final"

    @model_validator(mode="before")
    @classmethod
    def sin_valores_vacios(cls, valores):
        """Una variable declarada y vacía vale como no declarada.

        `.env.example` trae vacías las opcionales, que es la forma de mostrar que existen. Sin
        esto, copiarlo tal cual y completar solo la clave deja `TEMPERATURA=` como cadena vacía
        y el sistema no arranca; peor todavía, `DIRECTORIO_VECTORSTORE=` daría `Path('.')` y el
        índice se reconstruiría en cada arranque sin que nadie sepa por qué.
        """
        if isinstance(valores, dict):
            return {k: v for k, v in valores.items()
                    if not (isinstance(v, str) and not v.strip())}
        return valores

    def clave_de(self, proveedor: Proveedor) -> SecretStr:
        """La clave del proveedor, o un error que dice cuál falta y dónde cargarla."""
        clave = getattr(self, f"{proveedor.value}_api_key", None)
        if clave is None or not clave.get_secret_value().strip():
            raise ErrorDeConfiguracion(
                cfg.ERROR_CLAVE_DE_PROVEEDOR.format(variable=f"{proveedor.value.upper()}_API_KEY")
            )
        return clave


@lru_cache(maxsize=1)
def obtener_ajustes() -> Ajustes:
    """Los ajustes del proceso, leídos una sola vez."""
    return Ajustes()


def reiniciar_ajustes() -> None:
    """Borra la caché para que la próxima lectura vuelva a mirar el entorno."""
    obtener_ajustes.cache_clear()


def cargar_entorno() -> Ajustes:
    """Carga el `.env` y devuelve los ajustes ya validados.

    `find_dotenv` busca hacia arriba, así que en desarrollo encuentra el `.env` que vive
    fuera del repositorio y en el contenedor toma las variables que pasa el compose.

    `OPENAI_API_KEY` se exige siempre: los embeddings del índice son de OpenAI incluso
    cuando el chat corre con otro proveedor.
    """
    load_dotenv(find_dotenv(usecwd=True))
    reiniciar_ajustes()
    ajustes = obtener_ajustes()
    ajustes.clave_de(Proveedor.OPENAI)
    return ajustes

"""Los especialistas del orquestador.

Uno por dominio, y con acceso acotado: el investigador solo puede leer el corpus, el
verificador solo puede consultar el padron de citas, y el redactor solo puede pedir los links
de las citas que el verificador ya aprobo. Ninguno se llama al otro: eso lo decide el
supervisor.
"""

from app.agentes.analyst_agent import verificador_node
from app.agentes.research_agent import investigador_node
from app.agentes.writer_agent import redactor_node

__all__ = ["investigador_node", "verificador_node", "redactor_node"]

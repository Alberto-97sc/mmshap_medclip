# Dispara los registros de modelos y datasets al importar el paquete
from . import models      # registra modelos
from . import datasets    # registra datasets

# Módulo de comparación de modelos
from . import comparison  # funciones para comparar múltiples modelos

# (opcional) expón utilidades si quieres:
# from .registry import build_model, build_dataset

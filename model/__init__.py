"""
model/__init__.py
=================
Paquete de inferencia: preprocesamiento y predicción con el modelo CNN.
"""

from .preprocessor import AudioPreprocessor
from .inference import ModeloReconocedor

__all__ = ["AudioPreprocessor", "ModeloReconocedor"]

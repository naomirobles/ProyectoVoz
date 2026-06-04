"""
audio/__init__.py
=================
Paquete de captura y análisis de audio en tiempo real.
"""

from .capture import AudioCapture
from .vad import VoiceActivityDetector
from .buffer import AudioBuffer

__all__ = ["AudioCapture", "VoiceActivityDetector", "AudioBuffer"]

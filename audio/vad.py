"""
audio/vad.py
============
Detección de Actividad de Voz (Voice Activity Detection).

Usa un enfoque basado en energía RMS con histéresis:
  - Si la energía supera un umbral → hay voz (SPEAKING).
  - Si cae por debajo del umbral durante N frames → silencio (SILENCE).

La histéresis evita que el estado cambie constantemente en zonas
de transición ruidosas.

Estados posibles:
  'silence'   → no se detecta voz.
  'speaking'  → se está hablando.
  'end_word'  → se terminó de hablar (silencio tras voz detectada).
"""

import numpy as np
from collections import deque


class VoiceActivityDetector:
    """
    Detector de actividad de voz basado en energía RMS.

    Parámetros
    ----------
    sample_rate : int
        Frecuencia de muestreo del audio (Hz).
    energy_threshold : float
        Umbral de energía RMS. Si RMS > threshold → voz.
        Valor típico: 0.01–0.05 según ruido ambiental.
    silence_duration_s : float
        Segundos de silencio continuo para declarar fin de palabra.
        Por defecto 0.6 s — tiempo razonable entre palabras.
    min_speech_duration_s : float
        Duración mínima de voz para considerar que hay palabra válida
        (descarta clics y ruidos cortos).
    smoothing_frames : int
        Número de frames sobre los que se promedia la energía para
        suavizar transiciones bruscas.
    """

    # Constantes de estado
    SILENCE = "silence"
    SPEAKING = "speaking"
    END_WORD = "end_word"

    def __init__(
        self,
        sample_rate: int = 16_000,
        energy_threshold: float = 0.02,
        silence_duration_s: float = 0.6,
        min_speech_duration_s: float = 0.15,
        smoothing_frames: int = 5,
    ):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_duration_s = silence_duration_s
        self.min_speech_duration_s = min_speech_duration_s
        self.smoothing_frames = smoothing_frames

        # Estado interno
        self._state = self.SILENCE
        self._silence_frames = 0       # frames consecutivos de silencio
        self._speech_frames = 0        # frames de voz acumulados
        self._samples_per_frame = 512  # se actualiza al procesar

        # Buffer circular de energías para suavizado
        self._energy_buffer: deque = deque(maxlen=smoothing_frames)

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def energy_threshold(self) -> float:
        return self._threshold

    @energy_threshold.setter
    def energy_threshold(self, value: float) -> None:
        if value <= 0:
            raise ValueError("El umbral de energía debe ser positivo.")
        self._threshold = value

    # ------------------------------------------------------------------
    # Proceso principal
    # ------------------------------------------------------------------

    def process(self, chunk: np.ndarray) -> str:
        """
        Procesa un bloque de audio y actualiza el estado VAD.

        Parámetros
        ----------
        chunk : np.ndarray
            Array 1-D float32 de muestras de audio.

        Retorna
        -------
        str
            El estado actual: 'silence', 'speaking' o 'end_word'.
        """
        n = len(chunk)
        self._samples_per_frame = n

        # Calcular energía RMS del bloque
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        self._energy_buffer.append(rms)

        # Energía suavizada (media de los últimos N frames)
        smoothed_rms = float(np.mean(self._energy_buffer))

        # ---- Máquina de estados ----
        if smoothed_rms >= self._threshold:
            # Hay energía de voz
            self._silence_frames = 0
            self._speech_frames += 1
            self._state = self.SPEAKING

        else:
            # No hay energía suficiente
            if self._state == self.SPEAKING:
                self._silence_frames += 1

                # Calcular duración del silencio acumulado
                frames_para_fin = self._frames_para_segundos(self.silence_duration_s, n)

                if self._silence_frames >= frames_para_fin:
                    # Verificar que hubo suficiente voz previa
                    frames_minimos = self._frames_para_segundos(self.min_speech_duration_s, n)
                    if self._speech_frames >= frames_minimos:
                        self._state = self.END_WORD
                    else:
                        # Fue ruido corto, volver a silencio
                        self._reset()
            else:
                # Ya estábamos en silencio o END_WORD
                if self._state == self.END_WORD:
                    # El caller ya procesó este evento; resetear
                    self._reset()
                else:
                    self._state = self.SILENCE

        return self._state

    def reset_after_word(self) -> None:
        """
        Llamar después de haber procesado un END_WORD para
        reiniciar los contadores y volver a escuchar.
        """
        self._reset()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self._state = self.SILENCE
        self._silence_frames = 0
        self._speech_frames = 0
        self._energy_buffer.clear()

    def _frames_para_segundos(self, segundos: float, samples_per_frame: int) -> int:
        """Convierte segundos a número de frames."""
        samples_needed = int(self.sample_rate * segundos)
        return max(1, samples_needed // samples_per_frame)

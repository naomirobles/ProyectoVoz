"""
audio/buffer.py
===============
Buffer de audio para acumular muestras durante un segmento de voz.

Cuando el VAD detecta que se está hablando, el AudioBuffer
va acumulando los chunks de audio. Cuando el VAD señala
fin de palabra (END_WORD), el buffer entrega el audio
completo del segmento para su preprocesamiento.
"""

import numpy as np


class AudioBuffer:
    """
    Acumula bloques de audio en un buffer lineal.

    El buffer guarda también un 'pre-roll': un número configurable
    de frames anteriores al inicio de la voz, para no perder
    la consonante inicial de la palabra (que suele ser silenciosa
    o de baja energía).

    Parámetros
    ----------
    sample_rate : int
        Frecuencia de muestreo en Hz.
    pre_roll_s : float
        Segundos de audio anteriores al onset de voz que se incluyen
        en la grabación. Por defecto 0.1 s (≈ 1600 muestras).
    max_duration_s : float
        Duración máxima de acumulación. Si se supera, se trunca
        el principio para no crecer indefinidamente.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        pre_roll_s: float = 0.1,
        max_duration_s: float = 3.0,
    ):
        self.sample_rate = sample_rate
        self.pre_roll_samples = int(sample_rate * pre_roll_s)
        self.max_samples = int(sample_rate * max_duration_s)

        # Buffer principal: lista de arrays que se concatenarán al final
        self._chunks: list[np.ndarray] = []
        self._total_samples: int = 0

        # Pre-roll: cola circular de chunks recientes
        self._pre_roll_chunks: list[np.ndarray] = []
        self._pre_roll_samples: int = 0

        self._recording = False

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def push(self, chunk: np.ndarray) -> None:
        """
        Agrega un chunk al buffer.

        Si estamos grabando (recording=True) el chunk va al buffer
        principal. Si no, va al pre-roll circular.
        """
        if self._recording:
            self._add_to_main(chunk)
        else:
            self._add_to_pre_roll(chunk)

    def start_recording(self) -> None:
        """
        Activa la grabación e incorpora el pre-roll al buffer principal.
        Llamar cuando VAD detecta inicio de voz.
        """
        if self._recording:
            return  # ya estábamos grabando

        self._recording = True

        # Incorporar pre-roll
        if self._pre_roll_chunks:
            pre_roll_audio = np.concatenate(self._pre_roll_chunks)
            self._add_to_main(pre_roll_audio)

        # Limpiar pre-roll
        self._pre_roll_chunks = []
        self._pre_roll_samples = 0

    def stop_recording(self) -> np.ndarray:
        """
        Detiene la grabación y retorna el audio acumulado.
        Llamar cuando VAD detecta fin de palabra.

        Retorna
        -------
        np.ndarray
            Array 1-D float32 con todas las muestras del segmento.
            Puede tener longitud 0 si no se grabó nada.
        """
        self._recording = False

        if not self._chunks:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(self._chunks)
        self.clear()
        return audio

    def clear(self) -> None:
        """Vacía el buffer principal (pero no el pre-roll)."""
        self._chunks = []
        self._total_samples = 0

    def full_clear(self) -> None:
        """Vacía tanto el buffer principal como el pre-roll."""
        self.clear()
        self._pre_roll_chunks = []
        self._pre_roll_samples = 0
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def duration_s(self) -> float:
        """Duración en segundos del audio acumulado."""
        return self._total_samples / self.sample_rate

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _add_to_main(self, chunk: np.ndarray) -> None:
        """Agrega al buffer principal con límite de tamaño."""
        self._chunks.append(chunk.astype(np.float32))
        self._total_samples += len(chunk)

        # Si supera el máximo, recortar el principio
        if self._total_samples > self.max_samples:
            exceso = self._total_samples - self.max_samples
            self._trim_start(exceso)

    def _trim_start(self, n_samples: int) -> None:
        """Descarta las primeras n_samples muestras del buffer."""
        recortado = 0
        nuevos_chunks = []

        for chunk in self._chunks:
            if recortado >= n_samples:
                nuevos_chunks.append(chunk)
            elif recortado + len(chunk) <= n_samples:
                recortado += len(chunk)
            else:
                inicio = n_samples - recortado
                nuevos_chunks.append(chunk[inicio:])
                recortado = n_samples

        self._chunks = nuevos_chunks
        self._total_samples = sum(len(c) for c in self._chunks)

    def _add_to_pre_roll(self, chunk: np.ndarray) -> None:
        """Agrega al pre-roll circular, descartando lo viejo."""
        self._pre_roll_chunks.append(chunk.astype(np.float32))
        self._pre_roll_samples += len(chunk)

        # Descartar chunks viejos si supera el pre-roll
        while self._pre_roll_samples > self.pre_roll_samples and self._pre_roll_chunks:
            oldest = self._pre_roll_chunks.pop(0)
            self._pre_roll_samples -= len(oldest)

"""
audio/capture.py
================
Módulo de captura de audio en tiempo real desde el micrófono.

Utiliza sounddevice para abrir un stream de audio continuo.
El audio se entrega en bloques (chunks) a un callback,
que los deposita en una cola thread-safe para que el
procesador VAD los consuma sin bloquear la UI.
"""

import queue
import threading
import numpy as np
import sounddevice as sd


class AudioCapture:
    """
    Abre un stream de micrófono con sounddevice y deposita
    cada bloque de muestras en self.queue (queue.Queue).

    Parámetros
    ----------
    sample_rate : int
        Frecuencia de muestreo en Hz. Debe coincidir con la
        frecuencia esperada por el modelo (16 000 Hz).
    block_size : int
        Número de muestras por bloque entregado al callback.
        Un valor menor da menor latencia pero mayor CPU.
        512 muestras ≈ 32 ms a 16 kHz → buen balance.
    channels : int
        Siempre 1 (mono). El modelo fue entrenado con audio mono.
    """

    def __init__(self, sample_rate: int = 16_000, block_size: int = 512, channels: int = 1):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.channels = channels

        # Cola thread-safe donde se depositan los bloques de audio.
        # maxsize=0 significa cola ilimitada; se puede ajustar si
        # se quiere descartar bloques cuando el procesador es lento.
        self.queue: queue.Queue = queue.Queue()

        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self.is_running = False

    # ------------------------------------------------------------------
    # Callback interno de sounddevice
    # ------------------------------------------------------------------

    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time, status) -> None:
        """
        sounddevice llama a esta función cada vez que tiene 'frames'
        muestras listas.  Se ejecuta en un hilo interno de sounddevice,
        por eso sólo hacemos put_nowait (no bloqueante) para no
        ralentizarlo.

        indata tiene shape (frames, channels).
        Convertimos a float32 1-D antes de encolar.
        """
        if status:
            # Reportar overflows / underruns sin lanzar excepción
            print(f"[AudioCapture] Estado del stream: {status}")

        # Tomamos el primer canal (índice 0) → array 1-D
        mono = indata[:, 0].copy().astype(np.float32)
        try:
            self.queue.put_nowait(mono)
        except queue.Full:
            pass  # Si la cola está llena descartamos el bloque

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Abre el stream de micrófono y empieza a recibir audio."""
        with self._lock:
            if self.is_running:
                return

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=self.channels,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            self.is_running = True
            print("[AudioCapture] Stream iniciado.")

    def stop(self) -> None:
        """Cierra el stream de micrófono limpiamente."""
        with self._lock:
            if not self.is_running:
                return

            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None

            self.is_running = False
            print("[AudioCapture] Stream detenido.")

    def get_block(self, timeout: float = 0.1) -> np.ndarray | None:
        """
        Extrae un bloque de audio de la cola.

        Retorna None si no hay datos disponibles en 'timeout' segundos.
        """
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_queue(self) -> None:
        """Vacía la cola descartando todos los bloques pendientes."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

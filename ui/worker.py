"""
ui/worker.py
============
Hilo de trabajo (QThread) que orquesta la captura de audio,
el VAD y la inferencia del modelo.

Al separar este trabajo en un QThread evitamos bloquear el
hilo principal de Qt (que maneja la UI), lo que garantiza
que la interfaz siga respondiendo mientras se procesa audio.

Señales emitidas hacia la ventana principal:
  - estado_cambiado(str)    → "escuchando" | "hablando" | "procesando" | "detenido"
  - palabra_detectada(str, float) → (palabra, confianza)
  - error_ocurrido(str)     → mensaje de error para mostrar en UI
"""

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from audio.capture import AudioCapture
from audio.vad import VoiceActivityDetector
from audio.buffer import AudioBuffer
from model.preprocessor import AudioPreprocessor
from model.inference import ModeloReconocedor
from utils import config


class AudioWorker(QThread):
    """
    QThread que:
      1. Abre el stream de micrófono (AudioCapture).
      2. Procesa cada bloque con el VAD.
      3. Acumula audio durante la voz (AudioBuffer).
      4. Al detectar fin de palabra → preprocesa → infiere → emite señal.

    Parámetros
    ----------
    modelo : ModeloReconocedor
        Instancia ya cargada del modelo CNN.
    """

    # Señales Qt (deben declararse como atributos de clase)
    estado_cambiado = pyqtSignal(str)           # "escuchando", "hablando", etc.
    palabra_detectada = pyqtSignal(str, float)  # palabra, confianza
    error_ocurrido = pyqtSignal(str)            # mensaje de error

    def __init__(self, modelo: ModeloReconocedor, parent=None):
        super().__init__(parent)
        self.modelo = modelo

        # Instanciar componentes de audio
        self.captura = AudioCapture(
            sample_rate=config.SAMPLE_RATE,
            block_size=config.BLOCK_SIZE,
            channels=config.CHANNELS,
        )

        self.vad = VoiceActivityDetector(
            sample_rate=config.SAMPLE_RATE,
            energy_threshold=config.VAD_ENERGY_THRESHOLD,
            silence_duration_s=config.VAD_SILENCE_DURATION_S,
            min_speech_duration_s=config.VAD_MIN_SPEECH_DURATION_S,
            smoothing_frames=config.VAD_SMOOTHING_FRAMES,
        )

        self.buffer = AudioBuffer(
            sample_rate=config.SAMPLE_RATE,
            pre_roll_s=0.08,      # 80 ms de pre-roll
            max_duration_s=3.0,
        )

        self.preprocesador = AudioPreprocessor(
            sr_objetivo=config.SAMPLE_RATE,
            duracion_seg=config.DURACION_SEG,
            n_mels=config.N_MELS,
            time_steps=config.TIME_STEPS,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            win_length=config.WIN_LENGTH,
        )

        # Flag de control del bucle principal
        self._activo = False

    # ------------------------------------------------------------------
    # Control del hilo
    # ------------------------------------------------------------------

    def iniciar(self) -> None:
        """Inicia la captura de audio y el bucle de procesamiento."""
        self._activo = True
        self.start()  # Llama a run() en un hilo separado

    def detener(self) -> None:
        """Señala al hilo que debe terminar y espera a que lo haga."""
        self._activo = False
        self.captura.stop()
        self.wait(3000)  # Esperar hasta 3 segundos

    # ------------------------------------------------------------------
    # Bucle principal (se ejecuta en el hilo separado)
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Bucle principal del hilo.
        Se ejecuta automáticamente cuando se llama a self.start().
        """
        # Abrir stream de micrófono
        try:
            self.captura.start()
        except Exception as e:
            self.error_ocurrido.emit(f"Error al abrir el micrófono: {e}")
            return

        self.estado_cambiado.emit("escuchando")
        estado_anterior = VoiceActivityDetector.SILENCE

        while self._activo:
            # Obtener un bloque de audio de la cola
            chunk = self.captura.get_block(timeout=0.05)

            if chunk is None:
                # No hubo datos en el timeout, continuamos
                continue

            # ---- Siempre alimentar el buffer (para el pre-roll) ----
            self.buffer.push(chunk)

            # ---- Procesar con el VAD ----
            estado_vad = self.vad.process(chunk)

            # ---- Máquina de estados: reaccionar a cambios de estado ----

            if estado_vad == VoiceActivityDetector.SPEAKING:
                if estado_anterior != VoiceActivityDetector.SPEAKING:
                    # Transición silencio → voz: activar grabación
                    self.buffer.start_recording()
                    self.estado_cambiado.emit("hablando")

            elif estado_vad == VoiceActivityDetector.END_WORD:
                # Fin de segmento de voz → procesar audio
                self.estado_cambiado.emit("procesando")

                audio_segmento = self.buffer.stop_recording()
                self.vad.reset_after_word()

                if len(audio_segmento) > 0:
                    self._procesar_segmento(audio_segmento)

                self.estado_cambiado.emit("escuchando")

            elif estado_vad == VoiceActivityDetector.SILENCE:
                if estado_anterior == VoiceActivityDetector.SPEAKING:
                    # El VAD normalmente no va directo a SILENCE desde SPEAKING
                    # (pasa por END_WORD), pero por seguridad manejamos el caso.
                    pass

            estado_anterior = estado_vad

        # Limpieza al salir del bucle
        self.captura.stop()
        self.buffer.full_clear()
        self.estado_cambiado.emit("detenido")

    # ------------------------------------------------------------------
    # Procesamiento de un segmento de audio
    # ------------------------------------------------------------------

    def _procesar_segmento(self, audio: np.ndarray) -> None:
        """
        Preprocesa el segmento de audio y ejecuta la inferencia.

        Parámetros
        ----------
        audio : np.ndarray
            Array 1-D float32 con las muestras del segmento de voz.
        """
        try:
            # Convertir audio crudo → tensor Mel (1, 64, 64, 1)
            tensor = self.preprocesador.process(audio, sr_actual=config.SAMPLE_RATE)

            # Ejecutar inferencia en el modelo CNN
            palabra, confianza = self.modelo.predecir(tensor)

            # Emitir resultado hacia la UI
            self.palabra_detectada.emit(palabra, confianza)

        except Exception as e:
            print(f"[AudioWorker] Error procesando segmento: {e}")
            self.error_ocurrido.emit(f"Error de procesamiento: {e}")

    # ------------------------------------------------------------------
    # Ajustes en caliente (desde la UI)
    # ------------------------------------------------------------------

    def set_energy_threshold(self, valor: float) -> None:
        """Ajusta el umbral de energía VAD sin reiniciar el hilo."""
        self.vad.energy_threshold = valor

    def set_silence_duration(self, segundos: float) -> None:
        """Ajusta el tiempo de silencio para fin de palabra."""
        self.vad.silence_duration_s = segundos

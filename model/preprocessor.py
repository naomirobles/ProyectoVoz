"""
model/preprocessor.py
=====================
Preprocesamiento de audio para el modelo CNN de reconocimiento de palabras.

Replica EXACTAMENTE el pipeline usado durante el entrenamiento:
  1. Convertir a mono (ya viene mono del buffer).
  2. Remuestrear a 16 000 Hz si hace falta.
  3. Forzar duración de 1 segundo (padding con ceros o recorte).
  4. Generar espectrograma Mel (64 bandas, 64 pasos temporales).
  5. Convertir a dB y normalizar por muestra.
  6. Agregar dimensión de canal → shape (1, 64, 64, 1).

Si algún paso difiere del entrenamiento la accuracy caerá,
por eso se documentan los parámetros exactos.
"""

import numpy as np
import librosa


class AudioPreprocessor:
    """
    Convierte un segmento de audio crudo en el tensor de entrada
    esperado por el modelo CNN.

    Parámetros
    ----------
    sr_objetivo : int
        Sample rate objetivo (debe coincidir con el entrenamiento → 16 000).
    duracion_seg : float
        Duración fija en segundos (→ 1.0).
    n_mels : int
        Número de bandas Mel (→ 64).
    time_steps : int
        Ancho temporal del espectrograma (→ 64).
    n_fft : int
        Tamaño de la FFT (→ 1024).
    hop_length : int
        Desplazamiento entre ventanas (→ 256).
    win_length : int
        Tamaño de la ventana de análisis (→ 512).
    """

    def __init__(
        self,
        sr_objetivo: int = 16_000,
        duracion_seg: float = 1.0,
        n_mels: int = 64,
        time_steps: int = 64,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 512,
    ):
        self.sr_objetivo = sr_objetivo
        self.duracion_seg = duracion_seg
        self.n_mels = n_mels
        self.time_steps = time_steps
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # Longitud fija en muestras
        self.longitud_objetivo = int(sr_objetivo * duracion_seg)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def process(self, audio: np.ndarray, sr_actual: int | None = None) -> np.ndarray:
        """
        Pipeline completo: audio crudo → tensor para el modelo.

        Parámetros
        ----------
        audio : np.ndarray
            Array 1-D float32 de muestras de audio (mono esperado).
        sr_actual : int | None
            Sample rate del audio entrante. Si es None se asume
            que ya está a sr_objetivo.

        Retorna
        -------
        np.ndarray
            Tensor shape (1, n_mels, time_steps, 1) listo para model.predict().
        """
        signal = audio.astype(np.float32)

        # Paso 1: Si el audio viene en estéreo (no debería, pero por seguridad)
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)

        # Paso 2: Remuestrear si hace falta
        if sr_actual is not None and sr_actual != self.sr_objetivo:
            signal = librosa.resample(signal, orig_sr=sr_actual, target_sr=self.sr_objetivo)

        # Paso 3: Forzar duración fija
        signal = self._forzar_duracion(signal)

        # Paso 4: Espectrograma Mel → dB → normalización
        mel_db = self._extraer_mel(signal)

        # Paso 5: Agregar batch dimension y canal → (1, 64, 64, 1)
        tensor = mel_db[np.newaxis, ..., np.newaxis]

        return tensor.astype(np.float32)

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _forzar_duracion(self, signal: np.ndarray) -> np.ndarray:
        """Recorta o rellena con ceros hasta longitud_objetivo muestras."""
        largo = len(signal)

        if largo > self.longitud_objetivo:
            # Recortar: tomamos el centro del audio para conservar
            # la parte más energética (típicamente la vocal núcleo).
            inicio = (largo - self.longitud_objetivo) // 2
            signal = signal[inicio: inicio + self.longitud_objetivo]

        elif largo < self.longitud_objetivo:
            # Padding simétrico: mitad al inicio, mitad al final
            faltan = self.longitud_objetivo - largo
            pad_izq = faltan // 2
            pad_der = faltan - pad_izq
            signal = np.pad(signal, (pad_izq, pad_der), mode="constant")

        return signal

    def _extraer_mel(self, signal: np.ndarray) -> np.ndarray:
        """
        Genera el espectrograma Mel normalizado.

        Replica paso a paso el código de entrenamiento:
          librosa.feature.melspectrogram → power_to_db → ajuste temporal → norm.
        """
        mel = librosa.feature.melspectrogram(
            y=signal,
            sr=self.sr_objetivo,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=0,
            fmax=self.sr_objetivo // 2,
            power=2.0,
        )

        # Pasar a escala logarítmica dB
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Ajustar ancho temporal a time_steps
        if mel_db.shape[1] > self.time_steps:
            mel_db = mel_db[:, : self.time_steps]
        elif mel_db.shape[1] < self.time_steps:
            faltan = self.time_steps - mel_db.shape[1]
            valor_min = float(np.min(mel_db))
            mel_db = np.pad(
                mel_db,
                ((0, 0), (0, faltan)),
                mode="constant",
                constant_values=valor_min,
            )

        # Normalización por muestra (media 0, std 1)
        media = float(np.mean(mel_db))
        std = float(np.std(mel_db))
        mel_db = (mel_db - media) / (std + 1e-8)

        return mel_db.astype(np.float32)

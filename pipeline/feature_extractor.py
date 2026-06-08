"""
Extracción de espectrogramas Mel.

Expone:
  cargar_audio_fijo()      — carga y normaliza duración de un audio
  extraer_mel_fijo()       — convierte audio → espectrograma Mel (H×W)
  extraer_features_batch() — procesa un diccionario {palabra: [rutas]} → (X, y)
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa


def cargar_audio_fijo(
    ruta_wav: str,
    sr_objetivo: int = 16_000,
    duracion_seg: float = 1.0,
) -> Tuple[np.ndarray, int]:
    """
    Carga el audio, lo convierte a mono, lo remuestrea si hace falta
    y lo fuerza a exactamente `duracion_seg` segundos.
    """
    signal, sr = sf.read(ruta_wav)
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)
    if sr != sr_objetivo:
        signal = librosa.resample(signal.astype(np.float32),
                                  orig_sr=sr, target_sr=sr_objetivo)
        sr = sr_objetivo
    n_objetivo = int(sr_objetivo * duracion_seg)
    if len(signal) > n_objetivo:
        signal = signal[:n_objetivo]
    elif len(signal) < n_objetivo:
        signal = np.pad(signal, (0, n_objetivo - len(signal)), mode="constant")
    return signal.astype(np.float32), sr


def extraer_mel_fijo(
    ruta_wav: str,
    sr_objetivo: int = 16_000,
    duracion_seg: float = 1.0,
    n_mels: int = 64,
    time_steps: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 512,
) -> np.ndarray:
    """
    Devuelve espectrograma Mel de tamaño fijo (n_mels × time_steps),
    normalizado por muestra (media 0, std 1).
    """
    signal, sr = cargar_audio_fijo(ruta_wav, sr_objetivo=sr_objetivo,
                                   duracion_seg=duracion_seg)
    mel = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, n_mels=n_mels,
        fmin=0, fmax=sr // 2, power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Ajuste de ancho temporal
    if mel_db.shape[1] > time_steps:
        mel_db = mel_db[:, :time_steps]
    elif mel_db.shape[1] < time_steps:
        pad = time_steps - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)),
                        mode="constant", constant_values=np.min(mel_db))

    # Normalización por muestra
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)
    return mel_db.astype(np.float32)


def extraer_features_batch(
    ejemplos: Dict[str, List[str]],
    sr_objetivo: int,
    duracion_seg: float,
    n_mels: int,
    time_steps: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    max_por_palabra: Optional[int] = None,
    verbose_cada: int = 5_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrae espectrogramas para todas las palabras.

    Returns:
        X: array (N, n_mels, time_steps, 1)  — canal añadido para CNN
        y: array (N,) de etiquetas de texto
    """
    X_list, y_list = [], []
    errores = 0
    procesados = 0
    total = sum(
        min(len(r), max_por_palabra) if max_por_palabra else len(r)
        for r in ejemplos.values()
    )

    for palabra, rutas in ejemplos.items():
        if max_por_palabra:
            rutas = rutas[:max_por_palabra]
        for ruta in rutas:
            if not os.path.exists(ruta):
                errores += 1
                continue
            try:
                mel = extraer_mel_fijo(
                    ruta_wav=ruta,
                    sr_objetivo=sr_objetivo,
                    duracion_seg=duracion_seg,
                    n_mels=n_mels,
                    time_steps=time_steps,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                )
                X_list.append(mel)
                y_list.append(palabra)
            except Exception:
                errores += 1
            procesados += 1
            if verbose_cada and procesados % verbose_cada == 0:
                print(f"  [{procesados}/{total}] procesados | errores: {errores}")

    X = np.array(X_list)[..., np.newaxis]   # (N, H, W, 1)
    y = np.array(y_list)
    print(f"[FeatureExtractor] X={X.shape} | errores={errores}/{total}")
    return X, y

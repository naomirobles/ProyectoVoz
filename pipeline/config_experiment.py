"""
Configuración global del experimento.

Todos los parámetros y umbrales están aquí; no hay valores mágicos
dispersos por el código. Cada campo tiene un comentario que explica
qué justifica su valor por defecto.
"""

import os
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ExperimentConfig:
    # ── Reproducibilidad ──────────────────────────────────────────────
    seed: int = 42

    # ── Directorios ───────────────────────────────────────────────────
    output_dir: str = "outputs"
    models_dir: str = "models"

    # ── Dataset ───────────────────────────────────────────────────────
    dataset_name: str = "MLCommons/ml_spoken_words"
    dataset_lang: str = "es_wav"

    # ── Selección de palabras ─────────────────────────────────────────
    # Mínimo de muestras: por debajo de este valor el error de estimación
    # de accuracy por clase es demasiado alto (±5% con n=400, test_size=0.2).
    muestras_por_palabra: int = 1000
    max_palabras: int = 500
    # Palabras cortas (<= 4 chars) son fonéticamente ambiguas en español.
    min_longitud_palabra: int = 5
    # Umbral Jaro-Winkler para detectar pares fonéticamente similares.
    # 0.92 captura pares como "ciudad/ciudadano" sin ser demasiado agresivo.
    umbral_similitud_fonetica: float = 0.92

    # ── EDA: análisis de duración ─────────────────────────────────────
    # Número de audios muestreados por clase para estimar duración.
    # 50 por clase × 20 clases = 1000 archivos ≈ 30 s de análisis.
    muestras_analisis_duracion: int = 50

    # ── Filtrado por rendimiento (requiere entrenamiento preliminar) ───
    usar_filtrado_confusion: bool = False
    umbral_f1_preliminar: float = 0.25   # palabras con F1 < umbral → candidatas
    palabras_preliminar: int = 100       # palabras usadas en run preliminar
    muestras_preliminar: int = 200       # muestras/palabra en run preliminar

    # ── Parámetros de audio ───────────────────────────────────────────
    # SR 16 kHz: estándar de facto en reconocimiento de voz (Wav2Vec, DeepSpeech…)
    sr_objetivo: int = 16_000
    # 1 s cubre la duración modal de palabras aisladas en español.
    # El EDA confirma si es adecuado; ajustar según duracion_promedio_seg.
    duracion_seg: float = 1.0
    # 64 Mel-bands: compromiso entre resolución frecuencial y coste computacional.
    n_mels: int = 64
    # 64 time-steps: con hop=256 y SR=16k → 1 s ≈ 63 frames (se rellena a 64).
    time_steps: int = 64
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 512

    # ── Partición ─────────────────────────────────────────────────────
    test_size: float = 0.2    # 80/20 es estándar; con >400 clases garantiza
    val_size: float = 0.1     # representación por clase en val y test.

    # ── Entrenamiento ─────────────────────────────────────────────────
    max_epochs: int = 100     # techo alto; EarlyStopping decide cuándo parar.
    batch_size: int = 32
    patience_early_stop: int = 10
    patience_reduce_lr: int = 5
    factor_reduce_lr: float = 0.5

    # ── Keras Tuner ───────────────────────────────────────────────────
    ejecutar_tuner: bool = True
    # Subset pequeño para que el tuner sea viable en tiempo razonable.
    palabras_tuner: int = 50
    muestras_tuner: int = 300
    max_trials: int = 20
    tuner_epochs: int = 20    # épocas por trial durante la búsqueda

    # ── K-Fold ────────────────────────────────────────────────────────
    ejecutar_kfold: bool = True
    n_splits: int = 5         # 5-fold: estándar; balance entre varianza y coste.
    kfold_max_epochs: int = 30

    # ─────────────────────────────────────────────────────────────────

    def set_seeds(self) -> None:
        """Fija todas las fuentes de aleatoriedad para reproducibilidad."""
        random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        try:
            import numpy as np
            np.random.seed(self.seed)
        except ImportError:
            pass
        try:
            import tensorflow as tf
            tf.random.set_seed(self.seed)
        except ImportError:
            pass

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))

    def crear_directorios(self) -> None:
        for sub in ("eda", "tuner", "training", "validation", "interpretability"):
            Path(self.output_dir, sub).mkdir(parents=True, exist_ok=True)
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)

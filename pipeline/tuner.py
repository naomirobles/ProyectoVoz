"""
Búsqueda de hiperparámetros con Keras Tuner (Hyperband).

Espacio de búsqueda:
  Arquitectura : num_conv_layers, filters_i, kernel_i
  Regularización: dropout, batch_normalization, l2_reg
  Entrenamiento : learning_rate, optimizer

batch_size no es tunable dentro de Keras Tuner de forma nativa; se
compara en una grilla separada antes del tuner principal (ver
`buscar_batch_size`).

Produce:
  outputs/tuner/mejores_hiperparametros.json
  outputs/tuner/resumen_trials.csv
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# ── Modelo tunable ────────────────────────────────────────────────────

def _build_model(hp, num_clases: int, n_mels: int, time_steps: int) -> tf.keras.Model:
    num_conv  = hp.Int("num_conv_layers", 1, 4, default=3)
    dropout   = hp.Float("dropout", 0.1, 0.6, step=0.05, default=0.3)
    dense_u   = hp.Choice("dense_units", [64, 128, 256, 512], default=128)
    lr        = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3)
    use_bn    = hp.Boolean("batch_normalization", default=False)
    l2_val    = hp.Float("l2_reg", 0.0, 0.01, step=0.001, default=0.0)
    opt_name  = hp.Choice("optimizer", ["adam", "rmsprop"], default="adam")

    reg = regularizers.l2(l2_val) if l2_val > 0 else None

    inp = layers.Input(shape=(n_mels, time_steps, 1))
    x = inp
    for i in range(num_conv):
        filters = hp.Choice(f"filters_{i}", [16, 32, 64, 128], default=min(16 * 2**i, 128))
        kernel  = hp.Choice(f"kernel_{i}", [3, 5], default=3)
        x = layers.Conv2D(filters, (kernel, kernel), activation="relu",
                          padding="same", kernel_regularizer=reg)(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(dense_u, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_clases, activation="softmax")(x)

    model = models.Model(inp, out)
    opt_cfg = {"class_name": opt_name, "config": {"learning_rate": lr}}
    model.compile(
        optimizer=tf.keras.optimizers.get(opt_cfg),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Búsqueda de batch_size (grilla pequeña, previo al tuner) ──────────

def buscar_batch_size(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_clases: int,
    n_mels: int,
    time_steps: int,
    candidatos: Tuple[int, ...] = (16, 32, 64),
    epochs: int = 5,
    seed: int = 42,
) -> int:
    """
    Entrena el modelo por defecto con cada candidato de batch_size durante
    `epochs` épocas y devuelve el que logra mejor val_accuracy.
    Rápido: usa sólo `epochs` épocas y sin callbacks pesados.
    """
    print(f"[Tuner] Buscando batch_size óptimo entre {candidatos}...")
    mejor_bs = candidatos[0]
    mejor_acc = -1.0

    for bs in candidatos:
        tf.random.set_seed(seed)
        model = _construir_modelo_default(num_clases, n_mels, time_steps)
        hist = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=bs,
            verbose=0,
        )
        val_acc = max(hist.history["val_accuracy"])
        print(f"  batch_size={bs:3d} → val_accuracy={val_acc:.4f}")
        if val_acc > mejor_acc:
            mejor_acc = val_acc
            mejor_bs = bs
        tf.keras.backend.clear_session()

    print(f"[Tuner] Mejor batch_size: {mejor_bs} (val_acc={mejor_acc:.4f})")
    return mejor_bs


def _construir_modelo_default(num_clases, n_mels, time_steps) -> tf.keras.Model:
    inp = layers.Input(shape=(n_mels, time_steps, 1))
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_clases, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ── Tuner principal ───────────────────────────────────────────────────

def ejecutar_tuner(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_clases: int,
    n_mels: int,
    time_steps: int,
    output_dir: str,
    max_trials: int = 20,
    epochs: int = 20,
    batch_size: int = 32,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Ejecuta Hyperband y devuelve el dict de mejores hiperparámetros.
    Si keras_tuner no está instalado, devuelve {} y avisa.
    """
    try:
        import keras_tuner as kt
    except ImportError:
        print("[Tuner] keras_tuner no instalado. Ejecuta: pip install keras-tuner")
        print("[Tuner] Se usarán hiperparámetros por defecto.")
        return {}

    tuner_dir = Path(output_dir) / "tuner"
    tuner_dir.mkdir(parents=True, exist_ok=True)

    tuner = kt.Hyperband(
        hypermodel=lambda hp: _build_model(hp, num_clases, n_mels, time_steps),
        objective="val_accuracy",
        max_epochs=epochs,
        factor=3,
        seed=seed,
        directory=str(tuner_dir),
        project_name="cnn_voz_es",
        overwrite=True,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=3, factor=0.5, verbose=0),
    ]

    print(f"[Tuner] Iniciando Hyperband (max_trials≈{max_trials}, "
          f"epochs/trial={epochs}, batch={batch_size})...")
    tuner.search(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0].values

    # Guardar mejores HPs
    json_path = tuner_dir / "mejores_hiperparametros.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best_hps, f, ensure_ascii=False, indent=2)

    # Resumen de todos los trials
    _guardar_resumen_trials(tuner, tuner_dir)

    print(f"[Tuner] Mejores hiperparámetros:")
    for k, v in best_hps.items():
        print(f"  {k}: {v}")

    return best_hps


def _guardar_resumen_trials(tuner, tuner_dir: Path) -> None:
    try:
        filas = []
        for trial in tuner.oracle.trials.values():
            fila = {"trial_id": trial.trial_id,
                    "score": round(trial.score or 0.0, 4),
                    "status": trial.status}
            fila.update(trial.hyperparameters.values)
            filas.append(fila)

        if not filas:
            return

        campos = list(filas[0].keys())
        with open(tuner_dir / "resumen_trials.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
            w.writeheader()
            for fila in sorted(filas, key=lambda x: x.get("score", 0), reverse=True):
                w.writerow(fila)
        print(f"[Tuner] Resumen de trials → {tuner_dir}/resumen_trials.csv")
    except Exception as e:
        print(f"[Tuner] No se pudo guardar resumen de trials: {e}")

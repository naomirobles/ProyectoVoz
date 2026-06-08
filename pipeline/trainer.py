"""
Construcción del modelo y entrenamiento con análisis de épocas.

Analiza automáticamente:
  - Época óptima (máximo val_accuracy)
  - Punto de sobreajuste (val_loss aumenta consistentemente)
  - Ganancia marginal por época
  - Motivo de parada del EarlyStopping

Produce:
  outputs/training/historial_entrenamiento.csv
  outputs/training/historia.json
  outputs/training/curvas_entrenamiento.png
  outputs/training/analisis_epocas.json
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# ── Dataclass de resultado ────────────────────────────────────────────

@dataclass
class TrainingResult:
    accuracy_test: float
    loss_test: float
    epoch_optima: int
    total_epochs: int
    motivo_parada: str
    punto_overfitting: Optional[int]
    ganancia_marginal_ultima_epoca: float
    historia: Dict[str, List[float]] = field(default_factory=dict)


# ── Construcción del modelo ───────────────────────────────────────────

def construir_modelo(
    num_clases: int,
    n_mels: int,
    time_steps: int,
    hps: Optional[Dict[str, Any]] = None,
) -> tf.keras.Model:
    """
    Construye la CNN usando los hiperparámetros del tuner o valores por defecto.
    La arquitectura es idéntica a la que usa el tuner para que los pesos
    del mejor trial puedan cargarse sin cambios.
    """
    hps = hps or {}
    num_conv  = int(hps.get("num_conv_layers", 3))
    dropout   = float(hps.get("dropout", 0.3))
    dense_u   = int(hps.get("dense_units", 128))
    lr        = float(hps.get("learning_rate", 1e-3))
    use_bn    = bool(hps.get("batch_normalization", False))
    l2_val    = float(hps.get("l2_reg", 0.0))
    opt_name  = str(hps.get("optimizer", "adam"))

    reg = regularizers.l2(l2_val) if l2_val > 0 else None
    default_filters = [16, 32, 64, 128]

    inp = layers.Input(shape=(n_mels, time_steps, 1))
    x = inp
    for i in range(num_conv):
        f = int(hps.get(f"filters_{i}", default_filters[min(i, len(default_filters) - 1)]))
        k = int(hps.get(f"kernel_{i}", 3))
        x = layers.Conv2D(f, (k, k), activation="relu",
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


# ── Entrenamiento ─────────────────────────────────────────────────────

def entrenar_modelo(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str,
    max_epochs: int = 100,
    batch_size: int = 32,
    patience_early_stop: int = 10,
    patience_reduce_lr: int = 5,
    factor_reduce_lr: float = 0.5,
    val_size: float = 0.1,
) -> TrainingResult:
    """
    Entrena con EarlyStopping + ReduceLROnPlateau y analiza las curvas.
    """
    train_dir = Path(output_dir) / "training"
    train_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=patience_reduce_lr,
            factor=factor_reduce_lr,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(train_dir / "historial_entrenamiento.csv")),
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=val_size,
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    hist = history.history
    val_acc = hist["val_accuracy"]
    val_loss = hist["val_loss"]

    epoch_optima = int(np.argmax(val_acc)) + 1
    total_epochs = len(val_acc)
    punto_of = _detectar_overfitting(val_loss)
    ganancia = float(val_acc[-1] - val_acc[-2]) if len(val_acc) > 1 else 0.0

    if total_epochs < max_epochs:
        motivo = (f"EarlyStopping: sin mejora en val_accuracy "
                  f"durante {patience_early_stop} épocas consecutivas")
    else:
        motivo = f"Se alcanzó el máximo configurado de {max_epochs} épocas"

    # Evaluación en test (sin data leakage: test no participó en training)
    loss_test, acc_test = model.evaluate(X_test, y_test, verbose=0)

    # Guardar historia completa
    historia_serializable = {k: [float(v) for v in vals] for k, vals in hist.items()}
    with open(train_dir / "historia.json", "w", encoding="utf-8") as f:
        json.dump(historia_serializable, f, indent=2)

    # Análisis de épocas
    ganancias = [float(val_acc[i] - val_acc[i - 1]) for i in range(1, len(val_acc))]
    analisis = {
        "epoch_optima": epoch_optima,
        "total_epochs_ejecutadas": total_epochs,
        "max_epochs_configurado": max_epochs,
        "motivo_parada": motivo,
        "punto_overfitting_detectado": punto_of,
        "val_accuracy_en_epoch_optima": round(float(val_acc[epoch_optima - 1]), 4),
        "val_accuracy_final": round(float(val_acc[-1]), 4),
        "ganancia_marginal_ultima_epoca": round(ganancia, 4),
        "ganancia_media_por_epoca": round(float(np.mean(ganancias)), 4),
        "accuracy_test": round(float(acc_test), 4),
        "loss_test": round(float(loss_test), 4),
        "interpretacion": _interpretar_curvas(epoch_optima, total_epochs,
                                               punto_of, ganancia, max_epochs),
    }
    with open(train_dir / "analisis_epocas.json", "w", encoding="utf-8") as f:
        json.dump(analisis, f, ensure_ascii=False, indent=2)

    _graficar_curvas(hist, epoch_optima, punto_of, train_dir)

    print(f"\n[Trainer] Época óptima: {epoch_optima}/{total_epochs}")
    print(f"[Trainer] Motivo de parada: {motivo}")
    if punto_of:
        print(f"[Trainer] Sobreajuste detectado ~época {punto_of}")
    print(f"[Trainer] Accuracy test: {acc_test:.4f}")

    return TrainingResult(
        accuracy_test=float(acc_test),
        loss_test=float(loss_test),
        epoch_optima=epoch_optima,
        total_epochs=total_epochs,
        motivo_parada=motivo,
        punto_overfitting=punto_of,
        ganancia_marginal_ultima_epoca=ganancia,
        historia=historia_serializable,
    )


# ── Análisis de curvas ────────────────────────────────────────────────

def _detectar_overfitting(val_loss: List[float]) -> Optional[int]:
    """
    Devuelve la época (1-indexed) donde val_loss empieza a subir
    de forma consistente (3 épocas consecutivas al alza).
    """
    if len(val_loss) < 5:
        return None
    for i in range(1, len(val_loss) - 2):
        if val_loss[i] > val_loss[i - 1] and val_loss[i + 1] > val_loss[i]:
            return i + 1
    return None


def _interpretar_curvas(
    epoch_optima: int,
    total: int,
    overfitting: Optional[int],
    ganancia_marginal: float,
    max_epochs: int,
) -> str:
    partes = []
    partes.append(
        f"El modelo alcanzó su mejor val_accuracy en la época {epoch_optima} de {total} ejecutadas."
    )
    if overfitting:
        partes.append(
            f"Se detectó sobreajuste aproximadamente en la época {overfitting} "
            f"(val_loss comenzó a aumentar). EarlyStopping restauró los pesos de la época {epoch_optima}."
        )
    else:
        partes.append("No se detectó sobreajuste claro: val_loss fue estable o decreciente.")
    if abs(ganancia_marginal) < 0.001:
        partes.append(
            "La ganancia marginal en la última época es prácticamente nula (<0.1%), "
            "lo que confirma que el entrenamiento convergió."
        )
    if total < max_epochs:
        partes.append(
            f"El entrenamiento se detuvo {max_epochs - total} épocas antes del máximo "
            f"porque EarlyStopping no observó mejora."
        )
    return " ".join(partes)


# ── Graficado ─────────────────────────────────────────────────────────

def _graficar_curvas(
    hist: Dict[str, List[float]],
    epoch_optima: int,
    punto_of: Optional[int],
    output_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epocas = range(1, len(hist["accuracy"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for ax, metrica, titulo in [
            (ax1, "accuracy", "Accuracy"),
            (ax2, "loss", "Loss"),
        ]:
            ax.plot(epocas, hist[metrica], label=f"Train {titulo}", color="steelblue")
            ax.plot(epocas, hist[f"val_{metrica}"], label=f"Val {titulo}", color="darkorange")
            ax.axvline(epoch_optima, color="green", linestyle="--",
                       label=f"Época óptima ({epoch_optima})")
            if punto_of:
                ax.axvline(punto_of, color="red", linestyle=":",
                           label=f"Overfitting (~{punto_of})")
            ax.set_xlabel("Época")
            ax.set_ylabel(titulo)
            ax.set_title(f"Curvas de {titulo}")
            ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / "curvas_entrenamiento.png", dpi=120)
        plt.close()
        print(f"[Trainer] curvas_entrenamiento.png guardado.")
    except Exception as e:
        print(f"[Trainer] No se pudo graficar: {e}")

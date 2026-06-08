"""
Validación cruzada estratificada K-Fold con intervalos de confianza al 95%.

El K-Fold entrena el modelo desde cero en cada fold usando los mismos
hiperparámetros que el entrenamiento final, garantizando que la estimación
de accuracy refleje la varianza real del proceso de entrenamiento.

Produce:
  outputs/validation/kfold_resultados.json
  outputs/validation/kfold_por_fold.csv
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .trainer import construir_modelo


@dataclass
class ValidationResult:
    accuracy_mean: float
    accuracy_std: float
    accuracy_ci_95: Tuple[float, float]
    precision_macro_mean: float
    recall_macro_mean: float
    f1_macro_mean: float
    f1_macro_std: float
    resultados_por_fold: List[Dict] = field(default_factory=list)


def validacion_cruzada(
    X: np.ndarray,
    y_encoded: np.ndarray,
    num_clases: int,
    n_mels: int,
    time_steps: int,
    output_dir: str,
    n_splits: int = 5,
    max_epochs: int = 30,
    batch_size: int = 32,
    hps: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> ValidationResult:
    """
    Ejecuta StratifiedKFold y reporta métricas con intervalos de confianza.

    El intervalo de confianza del 95% usa la distribución t de Student
    (apropiada para muestras pequeñas como n_splits=5).
    """
    val_dir = Path(output_dir) / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    accs, precs, recs, f1s = [], [], [], []
    resultados_folds = []

    for fold, (idx_train, idx_val) in enumerate(skf.split(X, y_encoded), 1):
        print(f"\n[KFold] ── Fold {fold}/{n_splits} ──")
        X_tr, X_val = X[idx_train], X[idx_val]
        y_tr, y_val = y_encoded[idx_train], y_encoded[idx_val]

        model = construir_modelo(num_clases, n_mels, time_steps, hps)

        model.fit(
            X_tr, y_tr,
            validation_split=0.1,
            epochs=max_epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", patience=5, restore_best_weights=True)
            ],
            verbose=0,
        )

        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

        acc  = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
        rec  = recall_score(y_val, y_pred, average="macro", zero_division=0)
        f1   = f1_score(y_val, y_pred, average="macro", zero_division=0)

        accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1)

        fila = {
            "fold": fold,
            "accuracy": round(acc, 4),
            "precision_macro": round(prec, 4),
            "recall_macro": round(rec, 4),
            "f1_macro": round(f1, 4),
            "n_train": len(y_tr),
            "n_val": len(y_val),
        }
        resultados_folds.append(fila)
        print(f"  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")

        tf.keras.backend.clear_session()

    # ── Intervalo de confianza (t-Student 95%) ────────────────────────
    ci = _intervalo_confianza_95(accs)

    resultado = ValidationResult(
        accuracy_mean=round(float(np.mean(accs)), 4),
        accuracy_std=round(float(np.std(accs)), 4),
        accuracy_ci_95=(round(float(ci[0]), 4), round(float(ci[1]), 4)),
        precision_macro_mean=round(float(np.mean(precs)), 4),
        recall_macro_mean=round(float(np.mean(recs)), 4),
        f1_macro_mean=round(float(np.mean(f1s)), 4),
        f1_macro_std=round(float(np.std(f1s)), 4),
        resultados_por_fold=resultados_folds,
    )

    _guardar_resultados(resultado, val_dir)

    print(f"\n[KFold] Accuracy: {resultado.accuracy_mean:.4f} ± {resultado.accuracy_std:.4f}")
    print(f"[KFold] IC 95%:   [{resultado.accuracy_ci_95[0]:.4f}, "
          f"{resultado.accuracy_ci_95[1]:.4f}]")
    print(f"[KFold] F1 macro: {resultado.f1_macro_mean:.4f} ± {resultado.f1_macro_std:.4f}")

    return resultado


def _intervalo_confianza_95(valores: List[float]) -> Tuple[float, float]:
    """Intervalo de confianza al 95% usando distribución t de Student."""
    try:
        from scipy import stats
        n = len(valores)
        ci = stats.t.interval(
            0.95, df=n - 1,
            loc=np.mean(valores),
            scale=stats.sem(valores),
        )
        return float(ci[0]), float(ci[1])
    except ImportError:
        # Fallback: aproximación normal (válida para n≥30)
        media = np.mean(valores)
        sem = np.std(valores) / np.sqrt(len(valores))
        return float(media - 1.96 * sem), float(media + 1.96 * sem)


def _guardar_resultados(resultado: ValidationResult, val_dir: Path) -> None:
    resumen = {
        "accuracy_mean": resultado.accuracy_mean,
        "accuracy_std": resultado.accuracy_std,
        "accuracy_ci_95_nivel_0.95": list(resultado.accuracy_ci_95),
        "precision_macro_mean": resultado.precision_macro_mean,
        "recall_macro_mean": resultado.recall_macro_mean,
        "f1_macro_mean": resultado.f1_macro_mean,
        "f1_macro_std": resultado.f1_macro_std,
        "interpretacion_ci": (
            f"Con probabilidad del 95%, la accuracy real del modelo en datos no vistos "
            f"se encuentra entre {resultado.accuracy_ci_95[0]:.4f} y "
            f"{resultado.accuracy_ci_95[1]:.4f}."
        ),
        "folds": resultado.resultados_por_fold,
    }
    with open(val_dir / "kfold_resultados.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    csv_path = val_dir / "kfold_por_fold.csv"
    if resultado.resultados_por_fold:
        campos = list(resultado.resultados_por_fold[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            w.writerows(resultado.resultados_por_fold)

    print(f"[KFold] Resultados → {val_dir}/kfold_resultados.json")

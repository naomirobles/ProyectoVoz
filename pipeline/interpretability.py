"""
Análisis de interpretabilidad del modelo.

Responde:
  - ¿Qué clases son más fáciles de reconocer?
  - ¿Qué clases generan más errores?
  - ¿Cuáles son los pares más confundidos?
  - ¿Qué palabras deberían excluirse en un reentrenamiento?

Produce:
  outputs/interpretability/rendimiento_por_clase.csv
  outputs/interpretability/pares_confundidos.csv
  outputs/interpretability/resumen_interpretabilidad.json
  outputs/interpretability/matriz_confusion.png
  outputs/interpretability/f1_por_clase.png
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


@dataclass
class InterpretabilityResult:
    accuracy: float
    reporte_por_clase: Dict[str, Dict]
    top_mejores: List[Tuple[str, float]]    # (palabra, f1) mejores
    top_peores: List[Tuple[str, float]]     # (palabra, f1) peores
    pares_confundidos: List[Tuple[str, str, int]]  # (real, predicha, n_errores)


def analizar_rendimiento(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clases: List[str],
    output_dir: str,
    top_n: int = 15,
) -> InterpretabilityResult:
    """
    Genera análisis completo de rendimiento por clase y pares confundidos.
    """
    interp_dir = Path(output_dir) / "interpretability"
    interp_dir.mkdir(parents=True, exist_ok=True)

    # ── Reporte por clase ─────────────────────────────────────────────
    reporte_dict = classification_report(
        y_true, y_pred,
        target_names=clases,
        zero_division=0,
        output_dict=True,
    )
    metricas: Dict[str, Dict] = {}
    for clase in clases:
        if clase in reporte_dict:
            metricas[clase] = {
                "precision": round(reporte_dict[clase]["precision"], 4),
                "recall":    round(reporte_dict[clase]["recall"], 4),
                "f1":        round(reporte_dict[clase]["f1-score"], 4),
                "support":   int(reporte_dict[clase]["support"]),
            }

    # ── Top mejores / peores por F1 ───────────────────────────────────
    f1_scores = [(c, metricas[c]["f1"]) for c in clases if c in metricas]
    f1_scores.sort(key=lambda x: x[1], reverse=True)
    top_mejores = f1_scores[:top_n]
    top_peores  = list(reversed(f1_scores[-top_n:]))

    # ── Pares más confundidos ─────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    cm_sin_diag = cm.copy()
    np.fill_diagonal(cm_sin_diag, 0)

    pares: List[Tuple[str, str, int]] = []
    for i, j in zip(*np.where(cm_sin_diag > 0)):
        pares.append((clases[int(i)], clases[int(j)], int(cm_sin_diag[i, j])))
    pares.sort(key=lambda x: x[2], reverse=True)

    # ── Guardar CSV rendimiento por clase ─────────────────────────────
    csv_clase = interp_dir / "rendimiento_por_clase.csv"
    with open(csv_clase, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["palabra", "precision", "recall", "f1", "support", "categoria"])
        for clase in clases:
            if clase not in metricas:
                continue
            m = metricas[clase]
            if m["f1"] >= 0.80:
                cat = "excelente"
            elif m["f1"] >= 0.60:
                cat = "bueno"
            elif m["f1"] >= 0.40:
                cat = "regular"
            else:
                cat = "deficiente"
            w.writerow([clase, m["precision"], m["recall"], m["f1"], m["support"], cat])

    # ── Guardar CSV pares confundidos ─────────────────────────────────
    csv_conf = interp_dir / "pares_confundidos.csv"
    with open(csv_conf, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clase_real", "clase_predicha", "n_errores", "porcentaje_del_real"])
        for real, pred, n in pares[:100]:
            idx_real = clases.index(real)
            total_real = int(cm[idx_real].sum())
            pct = round(100 * n / total_real, 1) if total_real > 0 else 0.0
            w.writerow([real, pred, n, pct])

    # ── JSON de resumen ───────────────────────────────────────────────
    accuracy = float(np.mean(y_true == y_pred))
    macro_f1 = float(reporte_dict.get("macro avg", {}).get("f1-score", 0.0))

    resumen = {
        "accuracy_global": round(accuracy, 4),
        "f1_macro": round(macro_f1, 4),
        "n_clases": len(clases),
        "n_clases_f1_excelente_gt_0.8": sum(1 for _, f in f1_scores if f >= 0.80),
        "n_clases_f1_deficiente_lt_0.4": sum(1 for _, f in f1_scores if f < 0.40),
        "top_mejores_f1": [(c, round(f, 4)) for c, f in top_mejores],
        "top_peores_f1":  [(c, round(f, 4)) for c, f in top_peores],
        "top_20_pares_confundidos": [
            {"real": r, "predicha": p, "errores": n}
            for r, p, n in pares[:20]
        ],
    }
    with open(interp_dir / "resumen_interpretabilidad.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    # ── Gráficos ──────────────────────────────────────────────────────
    _graficar_f1_barras(top_mejores, top_peores, interp_dir)
    _graficar_matriz_confusion(y_true, y_pred, clases, top_peores, interp_dir)

    print(f"\n[Interpretability] Accuracy global: {accuracy:.4f}")
    print(f"[Interpretability] F1 macro: {macro_f1:.4f}")
    print(f"[Interpretability] Clases con F1≥0.80: "
          f"{sum(1 for _, f in f1_scores if f >= 0.80)}/{len(clases)}")
    print(f"[Interpretability] Top 5 mejores: {[c for c, _ in top_mejores[:5]]}")
    print(f"[Interpretability] Top 5 peores:  {[c for c, _ in top_peores[:5]]}")

    return InterpretabilityResult(
        accuracy=accuracy,
        reporte_por_clase=metricas,
        top_mejores=top_mejores,
        top_peores=top_peores,
        pares_confundidos=pares[:50],
    )


def detectar_palabras_problematicas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clases: List[str],
    umbral_f1: float = 0.25,
) -> Dict[str, float]:
    """
    Devuelve {palabra: f1} para palabras cuyo F1 está por debajo del umbral.
    Estas son candidatas a exclusión en un siguiente ciclo de entrenamiento.
    """
    from sklearn.metrics import f1_score
    f1_por_clase = f1_score(y_true, y_pred, average=None, zero_division=0)
    problematicas = {
        clases[i]: round(float(f1_por_clase[i]), 4)
        for i in range(len(clases))
        if i < len(f1_por_clase) and f1_por_clase[i] < umbral_f1
    }
    if problematicas:
        print(f"[Interpretability] {len(problematicas)} palabras con F1 < {umbral_f1}: "
              f"{list(problematicas.keys())[:10]}{'...' if len(problematicas) > 10 else ''}")
    return problematicas


# ── Gráficos ──────────────────────────────────────────────────────────

def _graficar_f1_barras(
    top_mejores: List[Tuple[str, float]],
    top_peores: List[Tuple[str, float]],
    output_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        nombres_m = [c for c, _ in top_mejores]
        f1_m = [f for _, f in top_mejores]
        ax1.barh(range(len(nombres_m)), f1_m, color="steelblue", edgecolor="white")
        ax1.set_yticks(range(len(nombres_m)))
        ax1.set_yticklabels(nombres_m, fontsize=9)
        ax1.set_xlabel("F1")
        ax1.set_xlim(0, 1.05)
        ax1.axvline(0.8, color="green", linestyle="--", linewidth=0.8, label="F1=0.80")
        ax1.set_title("Top clases — Mejor F1")
        ax1.legend(fontsize=8)

        nombres_p = [c for c, _ in top_peores]
        f1_p = [f for _, f in top_peores]
        ax2.barh(range(len(nombres_p)), f1_p, color="tomato", edgecolor="white")
        ax2.set_yticks(range(len(nombres_p)))
        ax2.set_yticklabels(nombres_p, fontsize=9)
        ax2.set_xlabel("F1")
        ax2.set_xlim(0, 1.05)
        ax2.axvline(0.4, color="red", linestyle="--", linewidth=0.8, label="F1=0.40")
        ax2.set_title("Top clases — Peor F1")
        ax2.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / "f1_por_clase.png", dpi=120)
        plt.close()
        print("[Interpretability] f1_por_clase.png guardado.")
    except Exception as e:
        print(f"[Interpretability] No se pudo graficar F1: {e}")


def _graficar_matriz_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clases: List[str],
    top_peores: List[Tuple[str, float]],
    output_dir: Path,
    n_clases_mostrar: int = 25,
) -> None:
    """Muestra la sub-matriz de confusión para las N clases con peor F1."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        peores_nombres = {c for c, _ in top_peores[:n_clases_mostrar]}
        idx_mostrar = [i for i, c in enumerate(clases) if c in peores_nombres]
        if not idx_mostrar:
            return

        mascara = np.isin(y_true, idx_mostrar)
        if mascara.sum() == 0:
            return

        y_t_sub = y_true[mascara]
        y_p_sub = y_pred[mascara]

        cm_sub = confusion_matrix(y_t_sub, y_p_sub, labels=idx_mostrar)
        etiquetas = [clases[i] for i in idx_mostrar]

        size = max(10, len(etiquetas) * 0.55)
        fig, ax = plt.subplots(figsize=(size, size * 0.9))
        im = ax.imshow(cm_sub, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        tick_fs = max(6, 10 - len(etiquetas) // 5)
        ax.set_xticks(range(len(etiquetas)))
        ax.set_yticks(range(len(etiquetas)))
        ax.set_xticklabels(etiquetas, rotation=45, ha="right", fontsize=tick_fs)
        ax.set_yticklabels(etiquetas, fontsize=tick_fs)
        ax.set_title(f"Matriz de Confusión — top {len(etiquetas)} clases con peor F1",
                     fontsize=11)
        ax.set_xlabel("Predicción"); ax.set_ylabel("Real")

        plt.tight_layout()
        plt.savefig(output_dir / "matriz_confusion.png", dpi=120)
        plt.close()
        print("[Interpretability] matriz_confusion.png guardado.")
    except Exception as e:
        print(f"[Interpretability] No se pudo graficar matriz: {e}")

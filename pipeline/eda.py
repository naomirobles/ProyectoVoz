"""
Análisis exploratorio del dataset (EDA).

Produce:
  outputs/eda/estadisticas.json          — métricas globales
  outputs/eda/frecuencias_palabras.csv   — muestras y longitud por palabra
  outputs/eda/histograma_frecuencias.png
  outputs/eda/histograma_duraciones.png
"""

import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class EDAResult:
    total_palabras: int
    total_muestras: int
    palabras_con_suficientes_muestras: int
    duracion_promedio_seg: float
    duracion_std_seg: float
    duracion_min_seg: float
    duracion_max_seg: float
    porcentaje_clases_balanceadas: float
    muestras_analizadas_duracion: int
    frecuencias: Dict[str, int]


def analizar_dataset(
    train_dataset,
    output_dir: str,
    muestras_por_palabra: int,
    muestras_analisis: int = 50,
    seed: int = 42,
) -> EDAResult:
    """
    Analiza el dataset sin cargar audio en memoria.
    La duración se estima leyendo sólo los headers de una muestra de archivos.
    """
    random.seed(seed)
    eda_dir = Path(output_dir) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Conteo de frecuencias ──────────────────────────────────────
    train_meta = train_dataset.remove_columns("audio")
    conteo: Counter = Counter(train_meta["keyword"])
    total_palabras = len(conteo)
    total_muestras = sum(conteo.values())
    palabras_suficientes = [p for p, c in conteo.items() if c >= muestras_por_palabra]

    # ── 2. CSV de frecuencias ─────────────────────────────────────────
    csv_path = eda_dir / "frecuencias_palabras.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["palabra", "muestras", "longitud_chars", "suficiente"])
        for palabra, count in conteo.most_common():
            w.writerow([palabra, count, len(palabra),
                        "si" if count >= muestras_por_palabra else "no"])
    print(f"[EDA] Frecuencias guardadas → {csv_path}")

    # ── 3. Análisis de duración (muestra aleatoria) ───────────────────
    palabras_muestra = random.sample(palabras_suficientes, min(20, len(palabras_suficientes)))
    set_muestra = set(palabras_muestra)

    indice: Dict[str, List[str]] = {p: [] for p in palabras_muestra}
    for fila in train_meta:
        if fila["keyword"] in set_muestra:
            indice[fila["keyword"]].append(fila["file"])

    duraciones: List[float] = []
    try:
        import soundfile as sf
        for palabra in palabras_muestra:
            archivos = indice[palabra]
            seleccion = random.sample(archivos, min(muestras_analisis, len(archivos)))
            for ruta in seleccion:
                try:
                    info = sf.info(ruta)
                    duraciones.append(info.duration)
                except Exception:
                    pass
    except ImportError:
        print("[EDA] soundfile no disponible; análisis de duración omitido.")

    if duraciones:
        arr = np.array(duraciones)
        dur_prom = float(np.mean(arr))
        dur_std  = float(np.std(arr))
        dur_min  = float(np.min(arr))
        dur_max  = float(np.max(arr))
    else:
        arr = np.array([1.0])
        dur_prom = dur_std = 0.0
        dur_min = dur_max = 0.0

    # ── 4. Balance de clases ──────────────────────────────────────────
    counts = np.array(list(conteo.values()), dtype=float)
    mediana = float(np.median(counts))
    pct_balanceadas = float(
        np.mean(np.abs(counts - mediana) / (mediana + 1e-8) <= 0.20) * 100
    )

    # ── 5. JSON de estadísticas ───────────────────────────────────────
    stats = {
        "total_palabras": total_palabras,
        "total_muestras": total_muestras,
        "palabras_con_suficientes_muestras": len(palabras_suficientes),
        "umbral_muestras_suficientes": muestras_por_palabra,
        "duracion_promedio_seg": round(dur_prom, 4),
        "duracion_std_seg": round(dur_std, 4),
        "duracion_min_seg": round(dur_min, 4),
        "duracion_max_seg": round(dur_max, 4),
        "porcentaje_clases_balanceadas": round(pct_balanceadas, 2),
        "nota_balance": "% de clases dentro del ±20% de la mediana de muestras",
        "muestras_analizadas_duracion": len(duraciones),
    }
    with open(eda_dir / "estadisticas.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # ── 6. Gráficos ───────────────────────────────────────────────────
    _graficar_frecuencias(conteo, eda_dir, muestras_por_palabra)
    if len(duraciones) > 5:
        _graficar_duraciones(arr, dur_prom, eda_dir)

    print(f"[EDA] Palabras totales: {total_palabras} | "
          f"Con ≥{muestras_por_palabra} muestras: {len(palabras_suficientes)}")
    print(f"[EDA] Duración: {dur_prom:.3f}s ± {dur_std:.3f}s  "
          f"[{dur_min:.3f}s – {dur_max:.3f}s]")
    print(f"[EDA] Balance: {pct_balanceadas:.1f}% de clases dentro del ±20% de la mediana")

    return EDAResult(
        total_palabras=total_palabras,
        total_muestras=total_muestras,
        palabras_con_suficientes_muestras=len(palabras_suficientes),
        duracion_promedio_seg=dur_prom,
        duracion_std_seg=dur_std,
        duracion_min_seg=dur_min,
        duracion_max_seg=dur_max,
        porcentaje_clases_balanceadas=pct_balanceadas,
        muestras_analizadas_duracion=len(duraciones),
        frecuencias=dict(conteo.most_common()),
    )


def _graficar_frecuencias(conteo: Counter, eda_dir: Path, umbral: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        counts = sorted(conteo.values(), reverse=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(len(counts)), counts, color="steelblue", alpha=0.7, width=1.0)
        ax.axhline(umbral, color="red", linestyle="--", linewidth=1.5,
                   label=f"Umbral mínimo ({umbral} muestras)")
        ax.set_xlabel("Palabras (orden descendente de frecuencia)")
        ax.set_ylabel("Número de muestras")
        ax.set_title("Distribución de muestras por palabra")
        ax.legend()
        plt.tight_layout()
        plt.savefig(eda_dir / "histograma_frecuencias.png", dpi=120)
        plt.close()
        print("[EDA] histograma_frecuencias.png guardado.")
    except Exception as e:
        print(f"[EDA] No se pudo graficar frecuencias: {e}")


def _graficar_duraciones(duraciones: np.ndarray, media: float, eda_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(duraciones, bins=40, color="darkorange", alpha=0.75, edgecolor="black")
        ax.axvline(media, color="red", linestyle="--", linewidth=1.5,
                   label=f"Media ({media:.3f} s)")
        ax.set_xlabel("Duración (segundos)")
        ax.set_ylabel("Frecuencia")
        ax.set_title("Distribución de duración de audios (muestra)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(eda_dir / "histograma_duraciones.png", dpi=120)
        plt.close()
        print("[EDA] histograma_duraciones.png guardado.")
    except Exception as e:
        print(f"[EDA] No se pudo graficar duraciones: {e}")

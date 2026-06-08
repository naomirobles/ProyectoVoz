"""
Selección automática de palabras con justificación cuantitativa.

Reemplaza la lista manual `palabrasDiscriminadas`.
Cada exclusión queda registrada con:
  - motivo_exclusion : categoría del filtro que la rechazó
  - metrica_exclusion: valor observado de la métrica
  - umbral_exclusion : umbral configurado
  - similar_a        : palabra con la que conflicta (si aplica)

Produce:
  outputs/eda/seleccion_palabras.csv
  outputs/eda/criterios_seleccion.json
"""

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class WordRecord:
    palabra: str
    muestras: int
    longitud: int
    seleccionada: bool
    motivo_exclusion: Optional[str] = None
    metrica_exclusion: Optional[float] = None
    umbral_exclusion: Optional[float] = None
    similar_a: Optional[str] = None
    similitud_fonetica: Optional[float] = None
    f1_preliminar: Optional[float] = None


def seleccionar_palabras(
    conteo: Counter,
    max_palabras: int,
    muestras_por_palabra: int,
    min_longitud: int,
    umbral_similitud: float,
    output_dir: str,
    palabras_problematicas: Optional[Dict[str, float]] = None,
    umbral_f1: float = 0.25,
) -> Tuple[List[str], List[WordRecord]]:
    """
    Pipeline de selección en 4 etapas:
      1. Filtro de frecuencia mínima (garantía estadística).
      2. Filtro de longitud mínima (palabras muy cortas son ambiguas).
      3. Filtro de rendimiento preliminar (si se proveen f1 por clase).
      4. Filtro de similitud fonética (evitar pares confundibles).

    Devuelve (palabras_seleccionadas, registros_detallados).
    """
    registros: List[WordRecord] = []

    # ── Etapa 1: frecuencia ───────────────────────────────────────────
    tras_frecuencia: Dict[str, int] = {}
    for palabra, count in conteo.items():
        if count < muestras_por_palabra:
            registros.append(WordRecord(
                palabra=palabra, muestras=count, longitud=len(palabra),
                seleccionada=False,
                motivo_exclusion="frecuencia_insuficiente",
                metrica_exclusion=float(count),
                umbral_exclusion=float(muestras_por_palabra),
            ))
        else:
            tras_frecuencia[palabra] = count

    # ── Etapa 2: longitud ─────────────────────────────────────────────
    tras_longitud: Dict[str, int] = {}
    for palabra, count in tras_frecuencia.items():
        if len(palabra) < min_longitud:
            registros.append(WordRecord(
                palabra=palabra, muestras=count, longitud=len(palabra),
                seleccionada=False,
                motivo_exclusion="longitud_insuficiente",
                metrica_exclusion=float(len(palabra)),
                umbral_exclusion=float(min_longitud),
            ))
        else:
            tras_longitud[palabra] = count

    # ── Etapa 3: rendimiento preliminar (opcional) ────────────────────
    tras_rendimiento: Dict[str, int] = {}
    for palabra, count in tras_longitud.items():
        f1 = palabras_problematicas.get(palabra) if palabras_problematicas else None
        if f1 is not None and f1 < umbral_f1:
            registros.append(WordRecord(
                palabra=palabra, muestras=count, longitud=len(palabra),
                seleccionada=False,
                motivo_exclusion="bajo_rendimiento_preliminar",
                metrica_exclusion=round(f1, 4),
                umbral_exclusion=umbral_f1,
                f1_preliminar=round(f1, 4),
            ))
        else:
            tras_rendimiento[palabra] = count

    # ── Etapa 4: similitud fonética ───────────────────────────────────
    # Ordenar por frecuencia descendente: la más frecuente gana en caso de conflicto.
    ordenadas = sorted(tras_rendimiento, key=tras_rendimiento.get, reverse=True)

    seleccionadas: List[str] = []
    excluidas_fon: Dict[str, Tuple[str, float]] = {}

    for palabra in ordenadas:
        if len(seleccionadas) >= max_palabras:
            break
        conflicto = _conflicto_fonetico(palabra, seleccionadas, umbral_similitud)
        if conflicto:
            similar_a, sim = conflicto
            excluidas_fon[palabra] = (similar_a, sim)
        else:
            seleccionadas.append(palabra)

    # Registros para palabras que pasaron etapas 1-3
    for palabra in ordenadas:
        count = tras_rendimiento[palabra]
        f1 = (palabras_problematicas or {}).get(palabra)
        if palabra in seleccionadas:
            registros.append(WordRecord(
                palabra=palabra, muestras=count, longitud=len(palabra),
                seleccionada=True,
                f1_preliminar=round(f1, 4) if f1 is not None else None,
            ))
        elif palabra in excluidas_fon:
            similar_a, sim = excluidas_fon[palabra]
            registros.append(WordRecord(
                palabra=palabra, muestras=count, longitud=len(palabra),
                seleccionada=False,
                motivo_exclusion="similitud_fonetica_alta",
                metrica_exclusion=round(sim, 4),
                umbral_exclusion=umbral_similitud,
                similar_a=similar_a,
                similitud_fonetica=round(sim, 4),
                f1_preliminar=round(f1, 4) if f1 is not None else None,
            ))

    _guardar_reporte(registros, seleccionadas, output_dir,
                     muestras_por_palabra, min_longitud, umbral_similitud, umbral_f1)

    n_frec = sum(1 for r in registros if r.motivo_exclusion == "frecuencia_insuficiente")
    n_lon  = sum(1 for r in registros if r.motivo_exclusion == "longitud_insuficiente")
    n_fon  = len(excluidas_fon)
    n_rend = sum(1 for r in registros if r.motivo_exclusion == "bajo_rendimiento_preliminar")

    print(f"[WordSelector] Seleccionadas: {len(seleccionadas)}")
    print(f"  Excluidas por frecuencia:      {n_frec}")
    print(f"  Excluidas por longitud:        {n_lon}")
    print(f"  Excluidas por similitud fon.:  {n_fon}")
    print(f"  Excluidas por rendimiento:     {n_rend}")

    return seleccionadas, registros


# ── Similitud fonética ────────────────────────────────────────────────

def _conflicto_fonetico(
    palabra: str,
    ya_seleccionadas: List[str],
    umbral: float,
) -> Optional[Tuple[str, float]]:
    """
    Detecta si 'palabra' es fonéticamente similar a alguna ya seleccionada.
    Usa jellyfish (Jaro-Winkler) si está instalado; sino, implementación propia.
    El español tiene correspondencia grafema-fonema alta, por lo que la
    similitud de cadenas es un proxy razonable para similitud fonética.
    """
    sim_fn = _jaro_winkler_jellyfish if _jellyfish_disponible() else _jaro_winkler_manual
    for existente in ya_seleccionadas:
        sim = sim_fn(palabra, existente)
        if sim >= umbral:
            return existente, sim
    return None


def _jellyfish_disponible() -> bool:
    try:
        import jellyfish  # noqa: F401
        return True
    except ImportError:
        return False


def _jaro_winkler_jellyfish(a: str, b: str) -> float:
    import jellyfish
    return jellyfish.jaro_winkler_similarity(a, b)


def _jaro_winkler_manual(a: str, b: str) -> float:
    """Implementación de Jaro-Winkler (fallback sin jellyfish)."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    rango = max(la, lb) // 2 - 1
    if rango < 0:
        rango = 0
    ma = [False] * la
    mb = [False] * lb
    matches = 0
    for i in range(la):
        ini = max(0, i - rango)
        fin = min(i + rango + 1, lb)
        for j in range(ini, fin):
            if not mb[j] and a[i] == b[j]:
                ma[i] = mb[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    seq_a = [a[i] for i in range(la) if ma[i]]
    seq_b = [b[j] for j in range(lb) if mb[j]]
    t = sum(1 for x, y in zip(seq_a, seq_b) if x != y) / 2
    jaro = (matches / la + matches / lb + (matches - t) / matches) / 3
    prefijo = sum(1 for i in range(min(4, la, lb)) if a[i] == b[i])
    return jaro + prefijo * 0.1 * (1 - jaro)


# ── Guardado ──────────────────────────────────────────────────────────

def _guardar_reporte(
    registros: List[WordRecord],
    seleccionadas: List[str],
    output_dir: str,
    muestras_min: int,
    longitud_min: int,
    umbral_sim: float,
    umbral_f1: float,
) -> None:
    eda_dir = Path(output_dir) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    campos = ["palabra", "muestras", "longitud", "seleccionada",
              "motivo_exclusion", "metrica_exclusion", "umbral_exclusion",
              "similar_a", "similitud_fonetica", "f1_preliminar"]

    csv_path = eda_dir / "seleccion_palabras.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for r in sorted(registros, key=lambda x: (not x.seleccionada, x.palabra)):
            w.writerow({
                "palabra": r.palabra,
                "muestras": r.muestras,
                "longitud": r.longitud,
                "seleccionada": "si" if r.seleccionada else "no",
                "motivo_exclusion": r.motivo_exclusion or "",
                "metrica_exclusion": r.metrica_exclusion if r.metrica_exclusion is not None else "",
                "umbral_exclusion": r.umbral_exclusion if r.umbral_exclusion is not None else "",
                "similar_a": r.similar_a or "",
                "similitud_fonetica": r.similitud_fonetica if r.similitud_fonetica is not None else "",
                "f1_preliminar": r.f1_preliminar if r.f1_preliminar is not None else "",
            })

    criterios = {
        "muestras_minimas": muestras_min,
        "longitud_minima_chars": longitud_min,
        "umbral_similitud_fonetica_jaro_winkler": umbral_sim,
        "umbral_f1_minimo_rendimiento_preliminar": umbral_f1,
        "total_seleccionadas": len(seleccionadas),
        "palabras_seleccionadas": seleccionadas,
        "nota_similitud": (
            "Usa jellyfish.jaro_winkler_similarity si está instalado; "
            "sino, implementación manual. El español tiene alta correspondencia "
            "grafema-fonema, por lo que la similitud de cadenas es proxy válido."
        ),
    }
    with open(eda_dir / "criterios_seleccion.json", "w", encoding="utf-8") as f:
        json.dump(criterios, f, ensure_ascii=False, indent=2)

    print(f"[WordSelector] Reporte → {csv_path}")

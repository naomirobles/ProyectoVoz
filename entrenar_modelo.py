"""
entrenar_modelo.py — Orquestador del pipeline basado en evidencia.

Etapas
──────
  1.  Configuración y reproducibilidad
  2.  Carga del dataset (sólo metadatos, sin audio en RAM)
  3.  Análisis exploratorio (EDA) → outputs/eda/
  4.  Selección automática de palabras (sin lista manual)
  5.  Búsqueda de batch_size óptimo + Keras Tuner → outputs/tuner/
  6.  Extracción de features completa
  7.  Entrenamiento con análisis de épocas → outputs/training/
  8.  Análisis de interpretabilidad → outputs/interpretability/
  9.  [Opcional] Filtrado de palabras problemáticas
 10.  [Opcional] Validación cruzada K-Fold → outputs/validation/
 11.  Exportar modelo y reporte final

Dependencias nuevas
───────────────────
  pip install keras-tuner jellyfish matplotlib scipy

Dependencias existentes
───────────────────────
  tensorflow, librosa, soundfile, scikit-learn, datasets, numpy
"""

import os
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from pipeline.config_experiment import ExperimentConfig
from pipeline.eda import analizar_dataset
from pipeline.word_selector import seleccionar_palabras
from pipeline.feature_extractor import extraer_features_batch
from pipeline.tuner import buscar_batch_size, ejecutar_tuner
from pipeline.trainer import construir_modelo, entrenar_modelo
from pipeline.interpretability import analizar_rendimiento, detectar_palabras_problematicas
from pipeline.reporter import ReporteExperimento

# ══════════════════════════════════════════════════════════════════════
# ETAPA 1 — Configuración y reproducibilidad
# ══════════════════════════════════════════════════════════════════════

cfg = ExperimentConfig(
    seed=42,
    dataset_name="MLCommons/ml_spoken_words",
    dataset_lang="es_wav",

    # Selección de palabras — umbrales con justificación en config_experiment.py
    max_palabras=500,
    muestras_por_palabra=1000,
    min_longitud_palabra=5,
    umbral_similitud_fonetica=0.92,

    # Filtrado por rendimiento (más preciso pero más lento)
    usar_filtrado_confusion=False,
    umbral_f1_preliminar=0.25,

    # Parámetros de audio
    sr_objetivo=16_000,
    duracion_seg=1.0,
    n_mels=64,
    time_steps=64,
    n_fft=1024,
    hop_length=256,
    win_length=512,

    # Partición
    test_size=0.20,
    val_size=0.10,

    # Entrenamiento — EarlyStopping decide el número real de épocas
    max_epochs=100,
    batch_size=32,
    patience_early_stop=10,
    patience_reduce_lr=5,
    factor_reduce_lr=0.5,

    # Tuner — subset pequeño para tiempo razonable
    ejecutar_tuner=True,
    palabras_tuner=50,
    muestras_tuner=300,
    max_trials=20,
    tuner_epochs=20,

    # K-Fold
    ejecutar_kfold=True,
    n_splits=5,
    kfold_max_epochs=30,
)

cfg.set_seeds()
cfg.crear_directorios()
cfg.save(f"{cfg.output_dir}/config_experimento.json")

reporte = ReporteExperimento(cfg.output_dir)
reporte.agregar("config", {
    "seed": cfg.seed,
    "max_palabras": cfg.max_palabras,
    "muestras_por_palabra": cfg.muestras_por_palabra,
    "min_longitud_palabra": cfg.min_longitud_palabra,
    "umbral_similitud_fonetica": cfg.umbral_similitud_fonetica,
    "sr_objetivo": cfg.sr_objetivo,
    "duracion_seg": cfg.duracion_seg,
    "n_mels": cfg.n_mels,
    "time_steps": cfg.time_steps,
    "n_fft": cfg.n_fft,
    "hop_length": cfg.hop_length,
    "batch_size": cfg.batch_size,
})

print("=" * 60)
print("  RECONOCIMIENTO DE PALABRAS EN ESPAÑOL")
print("  Pipeline basado en evidencia experimental")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════
# ETAPA 2 — Carga del dataset
# ══════════════════════════════════════════════════════════════════════

from datasets import load_dataset

print("\n[1/10] Cargando dataset (sólo metadatos)...")
dataset = load_dataset(cfg.dataset_name, cfg.dataset_lang, trust_remote_code=True)
train_ds = dataset["train"]
print(f"  {len(train_ds):,} muestras disponibles")

# ══════════════════════════════════════════════════════════════════════
# ETAPA 3 — Análisis exploratorio
# ══════════════════════════════════════════════════════════════════════

print("\n[2/10] Análisis exploratorio del dataset...")
eda_result = analizar_dataset(
    train_dataset=train_ds,
    output_dir=cfg.output_dir,
    muestras_por_palabra=cfg.muestras_por_palabra,
    muestras_analisis=cfg.muestras_analisis_duracion,
    seed=cfg.seed,
)
reporte.agregar("eda", {
    "total_palabras": eda_result.total_palabras,
    "total_muestras": eda_result.total_muestras,
    "palabras_con_suficientes_muestras": eda_result.palabras_con_suficientes_muestras,
    "duracion_promedio_seg": eda_result.duracion_promedio_seg,
    "duracion_std_seg": eda_result.duracion_std_seg,
    "duracion_min_seg": eda_result.duracion_min_seg,
    "duracion_max_seg": eda_result.duracion_max_seg,
    "porcentaje_clases_balanceadas": eda_result.porcentaje_clases_balanceadas,
})

# ══════════════════════════════════════════════════════════════════════
# ETAPA 4 — Selección automática de palabras
# ══════════════════════════════════════════════════════════════════════

print("\n[3/10] Selección automática de palabras (sin lista manual)...")
train_meta = train_ds.remove_columns("audio")
conteo: Counter = Counter(train_meta["keyword"])

palabras_seleccionadas, registros_seleccion = seleccionar_palabras(
    conteo=conteo,
    max_palabras=cfg.max_palabras,
    muestras_por_palabra=cfg.muestras_por_palabra,
    min_longitud=cfg.min_longitud_palabra,
    umbral_similitud=cfg.umbral_similitud_fonetica,
    output_dir=cfg.output_dir,
)

if len(palabras_seleccionadas) < 2:
    raise ValueError(
        f"Sólo {len(palabras_seleccionadas)} palabras pasaron los filtros. "
        "Ajusta muestras_por_palabra, min_longitud_palabra o umbral_similitud_fonetica."
    )

conteos_exc = {
    "frecuencia_insuficiente":     sum(1 for r in registros_seleccion
                                       if r.motivo_exclusion == "frecuencia_insuficiente"),
    "longitud_insuficiente":       sum(1 for r in registros_seleccion
                                       if r.motivo_exclusion == "longitud_insuficiente"),
    "similitud_fonetica_alta":     sum(1 for r in registros_seleccion
                                       if r.motivo_exclusion == "similitud_fonetica_alta"),
    "bajo_rendimiento_preliminar": sum(1 for r in registros_seleccion
                                       if r.motivo_exclusion == "bajo_rendimiento_preliminar"),
}
reporte.agregar("seleccion_palabras", {
    "total_seleccionadas": len(palabras_seleccionadas),
    "excluidas_por_filtro": conteos_exc,
    "criterios": {
        "muestras_minimas": cfg.muestras_por_palabra,
        "longitud_minima_chars": cfg.min_longitud_palabra,
        "umbral_similitud_fonetica_jaro_winkler": cfg.umbral_similitud_fonetica,
    },
})

# Indexar rutas de audio
print("  Indexando rutas de audio...")
set_pal = set(palabras_seleccionadas)
ejemplos: dict = {}

for fila in train_meta:
    p = fila["keyword"]
    if p in set_pal:
        if p not in ejemplos:
            ejemplos[p] = []
        if len(ejemplos[p]) < cfg.muestras_por_palabra:
            ejemplos[p].append(fila["file"])
    if all(len(ejemplos.get(pw, [])) >= cfg.muestras_por_palabra
           for pw in palabras_seleccionadas):
        break

print(f"  Rutas indexadas para {len(ejemplos)}/{len(palabras_seleccionadas)} palabras")

# ══════════════════════════════════════════════════════════════════════
# ETAPA 5 — Búsqueda de hiperparámetros (Keras Tuner)
# ══════════════════════════════════════════════════════════════════════

mejores_hps: dict = {}

if cfg.ejecutar_tuner:
    print(f"\n[4/10] Keras Tuner "
          f"({cfg.palabras_tuner} palabras × {cfg.muestras_tuner} muestras)...")

    palabras_t = palabras_seleccionadas[:cfg.palabras_tuner]
    ejemplos_t = {p: ejemplos[p][:cfg.muestras_tuner] for p in palabras_t}

    print("  Extrayendo features del subset de tuning...")
    X_t, y_t_raw = extraer_features_batch(
        ejemplos=ejemplos_t,
        sr_objetivo=cfg.sr_objetivo, duracion_seg=cfg.duracion_seg,
        n_mels=cfg.n_mels, time_steps=cfg.time_steps,
        n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length,
    )
    enc_t = LabelEncoder()
    y_t = enc_t.fit_transform(y_t_raw)
    num_cl_t = len(enc_t.classes_)

    X_tr_t, X_va_t, y_tr_t, y_va_t = train_test_split(
        X_t, y_t, test_size=0.2, random_state=cfg.seed, stratify=y_t
    )

    mejor_bs = buscar_batch_size(
        X_tr_t, y_tr_t, num_cl_t, cfg.n_mels, cfg.time_steps,
        candidatos=(16, 32, 64), epochs=5, seed=cfg.seed,
    )
    mejores_hps = ejecutar_tuner(
        X_train=X_tr_t, y_train=y_tr_t,
        num_clases=num_cl_t, n_mels=cfg.n_mels, time_steps=cfg.time_steps,
        output_dir=cfg.output_dir,
        max_trials=cfg.max_trials, epochs=cfg.tuner_epochs,
        batch_size=mejor_bs, seed=cfg.seed,
    )
    if mejores_hps:
        mejores_hps["batch_size_optimo"] = mejor_bs

    reporte.agregar("tuner", mejores_hps)
    reporte.agregar("tuner_trials", cfg.max_trials)
else:
    print("\n[4/10] Tuner omitido — se usarán valores por defecto")

# ══════════════════════════════════════════════════════════════════════
# ETAPA 6 — Extracción de features completa
# ══════════════════════════════════════════════════════════════════════

print(f"\n[5/10] Extrayendo features "
      f"({len(palabras_seleccionadas)} × {cfg.muestras_por_palabra})...")

X, y_raw = extraer_features_batch(
    ejemplos=ejemplos,
    sr_objetivo=cfg.sr_objetivo, duracion_seg=cfg.duracion_seg,
    n_mels=cfg.n_mels, time_steps=cfg.time_steps,
    n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length,
)

if len(X) == 0:
    raise ValueError("No se pudieron procesar audios. Verifica las rutas del dataset.")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_raw)
num_clases = len(encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=cfg.test_size, random_state=cfg.seed, stratify=y_encoded,
)
print(f"  Train={X_train.shape} | Test={X_test.shape} | Clases={num_clases}")

# ══════════════════════════════════════════════════════════════════════
# ETAPA 7 — Entrenamiento completo
# ══════════════════════════════════════════════════════════════════════

print(f"\n[6/10] Entrenamiento (max {cfg.max_epochs} épocas, EarlyStopping)...")

batch_final = (
    int(mejores_hps.get("batch_size_optimo", cfg.batch_size))
    if mejores_hps else cfg.batch_size
)

model = construir_modelo(
    num_clases=num_clases,
    n_mels=cfg.n_mels,
    time_steps=cfg.time_steps,
    hps=mejores_hps if mejores_hps else None,
)
model.summary()

train_result = entrenar_modelo(
    model=model,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    output_dir=cfg.output_dir,
    max_epochs=cfg.max_epochs,
    batch_size=batch_final,
    patience_early_stop=cfg.patience_early_stop,
    patience_reduce_lr=cfg.patience_reduce_lr,
    factor_reduce_lr=cfg.factor_reduce_lr,
    val_size=cfg.val_size,
)
reporte.agregar("entrenamiento", {
    "accuracy_test": train_result.accuracy_test,
    "loss_test": train_result.loss_test,
    "epoch_optima": train_result.epoch_optima,
    "total_epochs": train_result.total_epochs,
    "motivo_parada": train_result.motivo_parada,
    "punto_overfitting": train_result.punto_overfitting,
    "ganancia_marginal_ultima_epoca": train_result.ganancia_marginal_ultima_epoca,
})

# ══════════════════════════════════════════════════════════════════════
# ETAPA 8 — Interpretabilidad
# ══════════════════════════════════════════════════════════════════════

print("\n[7/10] Análisis de interpretabilidad por clase...")
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

interp_result = analizar_rendimiento(
    y_true=y_test, y_pred=y_pred,
    clases=list(encoder.classes_),
    output_dir=cfg.output_dir,
)
reporte.agregar("interpretabilidad", {
    "accuracy_global": interp_result.accuracy,
    "top_5_mejores": interp_result.top_mejores[:5],
    "top_5_peores":  interp_result.top_peores[:5],
    "top_10_confusiones": interp_result.pares_confundidos[:10],
})

# ══════════════════════════════════════════════════════════════════════
# ETAPA 9 — Filtrado de palabras problemáticas (opcional)
# ══════════════════════════════════════════════════════════════════════

if cfg.usar_filtrado_confusion:
    print(f"\n[8/10] Filtrando palabras con F1 < {cfg.umbral_f1_preliminar}...")
    problematicas = detectar_palabras_problematicas(
        y_true=y_test, y_pred=y_pred,
        clases=list(encoder.classes_),
        umbral_f1=cfg.umbral_f1_preliminar,
    )
    if problematicas:
        palabras_v2, _ = seleccionar_palabras(
            conteo=conteo,
            max_palabras=cfg.max_palabras,
            muestras_por_palabra=cfg.muestras_por_palabra,
            min_longitud=cfg.min_longitud_palabra,
            umbral_similitud=cfg.umbral_similitud_fonetica,
            output_dir=cfg.output_dir,
            palabras_problematicas=problematicas,
            umbral_f1=cfg.umbral_f1_preliminar,
        )
        print(f"  Palabras depuradas: {len(palabras_v2)}")
        print("  Para reentrenar con lista depurada, ejecuta de nuevo con "
              "usar_filtrado_confusion=False.")
    else:
        print("  No se encontraron palabras problemáticas.")
else:
    print("\n[8/10] Filtrado por confusion omitido (usar_filtrado_confusion=False)")

# ══════════════════════════════════════════════════════════════════════
# ETAPA 10 — Validación cruzada K-Fold (opcional)
# ══════════════════════════════════════════════════════════════════════

if cfg.ejecutar_kfold:
    from pipeline.validator import validacion_cruzada

    print(f"\n[9/10] Validación cruzada {cfg.n_splits}-Fold "
          f"(max {cfg.kfold_max_epochs} épocas/fold)...")
    kfold_result = validacion_cruzada(
        X=X, y_encoded=y_encoded,
        num_clases=num_clases,
        n_mels=cfg.n_mels, time_steps=cfg.time_steps,
        output_dir=cfg.output_dir,
        n_splits=cfg.n_splits,
        max_epochs=cfg.kfold_max_epochs,
        batch_size=batch_final,
        hps=mejores_hps if mejores_hps else None,
        seed=cfg.seed,
    )
    reporte.agregar("kfold", {
        "n_folds": cfg.n_splits,
        "accuracy_mean": kfold_result.accuracy_mean,
        "accuracy_std": kfold_result.accuracy_std,
        "accuracy_ci_95_nivel_0.95": list(kfold_result.accuracy_ci_95),
        "precision_macro_mean": kfold_result.precision_macro_mean,
        "recall_macro_mean": kfold_result.recall_macro_mean,
        "f1_macro_mean": kfold_result.f1_macro_mean,
        "f1_macro_std": kfold_result.f1_macro_std,
    })
else:
    print("\n[9/10] K-Fold omitido (ejecutar_kfold=False)")

# ══════════════════════════════════════════════════════════════════════
# ETAPA 11 — Exportar modelo y reporte
# ══════════════════════════════════════════════════════════════════════

print("\n[10/10] Exportando modelo y reporte final...")

Path(cfg.models_dir).mkdir(exist_ok=True)

model.save(f"{cfg.models_dir}/modelo_cnn.h5")
with open(f"{cfg.models_dir}/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

audio_cfg_dict = {
    "sr_objetivo": cfg.sr_objetivo,
    "duracion_seg": cfg.duracion_seg,
    "n_mels": cfg.n_mels,
    "time_steps": cfg.time_steps,
    "n_fft": cfg.n_fft,
    "hop_length": cfg.hop_length,
    "win_length": cfg.win_length,
    "num_clases": num_clases,
    "clases": encoder.classes_.tolist(),
}
with open(f"{cfg.models_dir}/audio_config.json", "w", encoding="utf-8") as f:
    json.dump(audio_cfg_dict, f, ensure_ascii=False, indent=2)

print(f"  {cfg.models_dir}/modelo_cnn.h5")
print(f"  {cfg.models_dir}/label_encoder.pkl")
print(f"  {cfg.models_dir}/audio_config.json")

reporte.generar()

print("\n" + "=" * 60)
print("  PIPELINE COMPLETADO")
print(f"  Accuracy test:  {train_result.accuracy_test:.4f}")
if cfg.ejecutar_kfold:
    print(f"  K-Fold ({cfg.n_splits}):  "
          f"{kfold_result.accuracy_mean:.4f} ± {kfold_result.accuracy_std:.4f}  "
          f"IC95%=[{kfold_result.accuracy_ci_95[0]:.4f}, "
          f"{kfold_result.accuracy_ci_95[1]:.4f}]")
print(f"  Reporte:        {cfg.output_dir}/reporte_final.txt")
print("=" * 60)

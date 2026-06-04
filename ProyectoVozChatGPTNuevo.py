import os
from collections import Counter

import numpy as np
import soundfile as sf
import librosa

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, models

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["NUMBA_CACHE_DIR"] = r"C:\Users\spide\numba_cache"

# Parámetros del experimento
MAX_PALABRAS = 500          # número de clases
MUESTRAS_POR_PALABRA = 150 # audios por clase
SR_OBJETIVO = 16000        # sample rate fijo
DURACION_SEG = 1.0         # todos los audios se forzarán a 1 segundo
N_MELS = 64                # número de bandas Mel
TIME_STEPS = 64            # ancho fijo del espectrograma
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Palabras que quieres excluir
palabrasDiscriminadas = [
    "segunda", "muchas", "historia", "grandes", "fueron", "general",
    "familia", "escuela", "españa", "cuenta", "dentro", "contra",
    "comenzo", "central", "capital", "canción", "aunque", "algunos",
    "centro", "algunas", "encuentran", "nombre"
]

# ============================================================
# PARTE 1: CARGAR DATASET Y ELEGIR PALABRAS
# ============================================================

print("Cargando dataset...")
dataset = load_dataset("MLCommons/ml_spoken_words", "es_wav")
train = dataset["train"]
print("Dataset cargado.")
print(train)

conteo = Counter(train["keyword"])
print("Total de palabras distintas:", len(conteo))

PalabrasSeleccionadas = []

print("\nPalabras frecuentes:")
for palabra, cantidad in conteo.most_common(1000):
    if len(palabra) > 3:
        print(f"{palabra}: {cantidad}")

    if cantidad >= MUESTRAS_POR_PALABRA and len(palabra) > 5 and palabra not in palabrasDiscriminadas:
        PalabrasSeleccionadas.append(palabra)

PalabrasSeleccionadas = PalabrasSeleccionadas[:MAX_PALABRAS]

print("\nPalabras seleccionadas:")
print(PalabrasSeleccionadas)

if len(PalabrasSeleccionadas) < 2:
    raise ValueError("No se encontraron suficientes palabras para entrenar.")

# ============================================================
# PARTE 2: GUARDAR RUTAS DE AUDIO POR PALABRA
# ============================================================

train_sin_audio = train.remove_columns("audio")

Ejemplos = {}

for fila in train_sin_audio:
    palabra = fila["keyword"]

    if palabra in PalabrasSeleccionadas:
        if palabra not in Ejemplos:
            Ejemplos[palabra] = []

        if len(Ejemplos[palabra]) < MUESTRAS_POR_PALABRA:
            Ejemplos[palabra].append(fila["file"])

    if all(len(Ejemplos.get(p, [])) >= MUESTRAS_POR_PALABRA for p in PalabrasSeleccionadas):
        break

print("\nResumen de ejemplos guardados:")
for palabra in PalabrasSeleccionadas:
    print(f"{palabra}: {len(Ejemplos.get(palabra, []))} archivos")

# ============================================================
# PARTE 3: FUNCIONES PARA ESPECTROGRAMA MEL FIJO
# ============================================================

def cargar_audio_fijo(ruta_wav, sr_objetivo=16000, duracion_seg=1.0):
    """
    Carga un audio, lo convierte a mono, lo remuestrea si hace falta,
    y lo fuerza a una duración fija.
    
    Si el audio dura menos, se rellena con ceros.
    Si dura más, se recorta.
    """
    signal, sr = sf.read(ruta_wav)

    # Si está en estéreo, lo hacemos mono
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    # Remuestreo si hace falta
    if sr != sr_objetivo:
        signal = librosa.resample(signal.astype(np.float32), orig_sr=sr, target_sr=sr_objetivo)
        sr = sr_objetivo

    # Longitud objetivo en muestras
    longitud_objetivo = int(sr_objetivo * duracion_seg)

    # Recortar o rellenar
    if len(signal) > longitud_objetivo:
        signal = signal[:longitud_objetivo]
    elif len(signal) < longitud_objetivo:
        faltan = longitud_objetivo - len(signal)
        signal = np.pad(signal, (0, faltan), mode="constant")

    return signal, sr


def extraer_mel_fijo(ruta_wav, sr_objetivo=16000, duracion_seg=1.0, n_mels=64, time_steps=64):
    """
    Convierte un audio en un espectrograma Mel de tamaño fijo:
    (n_mels, time_steps)
    """
    signal, sr = cargar_audio_fijo(ruta_wav, sr_objetivo=sr_objetivo, duracion_seg=duracion_seg)

    # Espectrograma Mel
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        win_length=512,
        n_mels=n_mels,
        fmin=0,
        fmax=sr // 2,
        power=2.0
    )

    # Pasar a dB
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Ajustar el ancho temporal a time_steps
    # Si sobran columnas, recortamos. Si faltan, rellenamos con el valor mínimo.
    if mel_db.shape[1] > time_steps:
        mel_db = mel_db[:, :time_steps]
    elif mel_db.shape[1] < time_steps:
        faltan = time_steps - mel_db.shape[1]
        valor_min = np.min(mel_db)
        mel_db = np.pad(mel_db, ((0, 0), (0, faltan)), mode="constant", constant_values=valor_min)

    # Normalización simple por muestra
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)

    return mel_db.astype(np.float32)


# ============================================================
# PARTE 4: CONSTRUIR X E y
# ============================================================

X = []
y = []

print("\nExtrayendo espectrogramas Mel...")

for palabra, rutas in Ejemplos.items():
    for ruta in rutas:
        if not os.path.exists(ruta):
            print(f"Archivo no encontrado: {ruta}")
            continue

        try:
            mel = extraer_mel_fijo(
                ruta_wav=ruta,
                sr_objetivo=SR_OBJETIVO,
                duracion_seg=DURACION_SEG,
                n_mels=N_MELS,
                time_steps=TIME_STEPS
            )

            X.append(mel)
            y.append(palabra)

        except Exception as e:
            print(f"Error procesando {ruta}: {e}")

X = np.array(X)
y = np.array(y)

print("\nFormas iniciales:")
print("X.shape =", X.shape)  # esperado: (num_audios, N_MELS, TIME_STEPS)
print("y.shape =", y.shape)

if len(X) == 0:
    raise ValueError("No se pudieron procesar audios.")

# Agregamos un canal para CNN: (muestras, alto, ancho, canal)
X = X[..., np.newaxis]

print("\nForma para CNN:")
print("X.shape =", X.shape)

# ============================================================
# PARTE 5: ETIQUETAS Y PARTICIÓN
# ============================================================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print("\nTrain/Test:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

num_clases = len(np.unique(y_encoded))
print("Número de clases:", num_clases)

# ============================================================
# PARTE 6: MODELO CNN
# ============================================================

model = models.Sequential([
    layers.Input(shape=(N_MELS, TIME_STEPS, 1)),

    layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_clases, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nResumen del modelo:")
model.summary()

# ============================================================
# PARTE 7: ENTRENAMIENTO
# ============================================================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=25,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ============================================================
# PARTE 8: EVALUACIÓN
# ============================================================

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred)
print("\n================ RESULTADOS ================")
print(f"Accuracy: {acc:.4f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=0))

# ============================================================
# PARTE 9: PRUEBA INDIVIDUAL
# ============================================================

print("\nEjemplo de predicción individual:")
ruta_prueba = Ejemplos[PalabrasSeleccionadas[0]][0]

mel_prueba = extraer_mel_fijo(
    ruta_wav=ruta_prueba,
    sr_objetivo=SR_OBJETIVO,
    duracion_seg=DURACION_SEG,
    n_mels=N_MELS,
    time_steps=TIME_STEPS
)

x_prueba = mel_prueba[np.newaxis, ..., np.newaxis]  # shape: (1, N_MELS, TIME_STEPS, 1)

pred_probs = model.predict(x_prueba)
pred_num = np.argmax(pred_probs, axis=1)[0]
pred_texto = encoder.inverse_transform([pred_num])[0]

print("Ruta de prueba:", ruta_prueba)
print("Palabra real:", PalabrasSeleccionadas[0])
print("Palabra predicha:", pred_texto)


# ============================================================
# PARTE 10: GUARDAR MODELO Y ENCODER PARA LA APLICACIÓN
# ============================================================

import pickle
import os

os.makedirs("models", exist_ok=True)

# Guardar el modelo Keras
model.save("models/modelo_cnn.h5")
print("Modelo guardado en models/modelo_cnn.h5")

# Guardar el LabelEncoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
print("LabelEncoder guardado en models/label_encoder.pkl")
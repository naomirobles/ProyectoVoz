"""
utils/config.py
===============
Configuración centralizada de la aplicación.

Todos los parámetros ajustables están aquí para que sea fácil
modificarlos sin tocar el código de los módulos.

IMPORTANTE: Las rutas del modelo y el encoder deben apuntar a
los archivos que generó tu entrenamiento.
"""

import os

# ------------------------------------------------------------------
# Rutas de archivos del modelo
# ------------------------------------------------------------------

# Directorio base del proyecto (carpeta que contiene main.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Carpeta donde guardar el modelo y el encoder
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ruta al modelo entrenado (puede ser .h5 o carpeta SavedModel)
RUTA_MODELO = os.path.join(MODELS_DIR, "modelo_cnn.h5")

# Ruta al LabelEncoder serializado con pickle
RUTA_ENCODER = os.path.join(MODELS_DIR, "label_encoder.pkl")

# ------------------------------------------------------------------
# Parámetros de audio (deben coincidir EXACTAMENTE con el entrenamiento)
# ------------------------------------------------------------------

SAMPLE_RATE = 16_000        # Hz — frecuencia de muestreo objetivo
DURACION_SEG = 1.0          # segundos — duración fija del segmento
N_MELS = 64                 # bandas del espectrograma Mel
TIME_STEPS = 64             # columnas (ancho temporal) del espectrograma
N_FFT = 1024                # tamaño de la FFT
HOP_LENGTH = 256            # desplazamiento entre ventanas
WIN_LENGTH = 512            # tamaño de la ventana de análisis

# ------------------------------------------------------------------
# Parámetros de captura de audio
# ------------------------------------------------------------------

BLOCK_SIZE = 512            # muestras por bloque del stream (≈ 32 ms a 16 kHz)
CHANNELS = 1                # siempre mono

# ------------------------------------------------------------------
# Parámetros del VAD (Voice Activity Detection)
# ------------------------------------------------------------------

# Umbral de energía RMS. Aumentar si hay mucho ruido de fondo;
# disminuir si el micrófono es muy lejano o la voz muy suave.
VAD_ENERGY_THRESHOLD = 0.015

# Segundos de silencio continuo para declarar fin de palabra.
# Aumentar si las palabras tienen pausas internas largas.
VAD_SILENCE_DURATION_S = 0.65

# Duración mínima de voz para considerar que es una palabra real
# (descarta clics, golpes, ruidos cortos).
VAD_MIN_SPEECH_DURATION_S = 0.15

# Número de frames sobre los que se suaviza la energía RMS.
VAD_SMOOTHING_FRAMES = 5

# ------------------------------------------------------------------
# Parámetros de inferencia
# ------------------------------------------------------------------

# Probabilidad mínima de softmax para aceptar una predicción.
# Si la confianza es menor, se muestra "???" en lugar de la palabra.
UMBRAL_CONFIANZA = 0.35

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------

APP_TITLE = "Reconocimiento de Palabras en Español"
APP_WIDTH = 800
APP_HEIGHT = 620

# Colores del tema (modo oscuro con acento guinda/rojo profundo)
COLOR_FONDO = "#0e0e0e"
COLOR_PANEL = "#1a1a1a"
COLOR_ACENTO = "#c0392b"          # rojo carmín
COLOR_ACENTO_SUAVE = "#922b21"
COLOR_TEXTO = "#e8e8e8"
COLOR_TEXTO_GRIS = "#888888"
COLOR_SPEAKING = "#27ae60"        # verde — hablando
COLOR_PROCESSING = "#e67e22"      # naranja — procesando
COLOR_LISTENING = "#3498db"       # azul — escuchando
COLOR_IDLE = "#555555"            # gris — detenido

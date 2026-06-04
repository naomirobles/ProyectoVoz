# Reconocimiento de Palabras en Español — Tiempo Real

Aplicación de escritorio PyQt6 que usa un modelo CNN entrenado con
espectrogramas Mel para reconocer palabras en español desde el micrófono.

---

## Estructura del proyecto

```
speech_recognizer/
│
├── main.py                  ← Punto de entrada
├── guardar_modelo.py        ← Script auxiliar para exportar el modelo
├── requirements.txt
│
├── models/                  ← Archivos del modelo (DEBES crearlos tú)
│   ├── modelo_cnn.h5        ← Modelo Keras entrenado
│   └── label_encoder.pkl    ← LabelEncoder de scikit-learn
│
├── audio/
│   ├── __init__.py
│   ├── capture.py           ← Stream de micrófono (sounddevice)
│   ├── vad.py               ← Detección de actividad de voz (energía RMS)
│   └── buffer.py            ← Acumulador de audio durante la voz
│
├── model/
│   ├── __init__.py
│   ├── preprocessor.py      ← Audio crudo → tensor Mel (64×64×1)
│   └── inference.py         ← Carga del modelo y predicción
│
├── ui/
│   ├── __init__.py
│   ├── main_window.py       ← Ventana principal PyQt6
│   └── worker.py            ← QThread de captura + VAD + inferencia
│
└── utils/
    ├── __init__.py
    └── config.py            ← Todos los parámetros configurables
```

---

## Instalación paso a paso

### 1. Crear y activar entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

> **Nota GPU**: Si tienes GPU NVIDIA y quieres aceleración, usa
> `tensorflow[and-cuda]` en lugar de `tensorflow` en requirements.txt.

### 3. Exportar el modelo entrenado

Al final de tu script de entrenamiento, agrega estas dos líneas
**después** de que `model` y `encoder` estén definidos:

```python
import os, pickle
os.makedirs("models", exist_ok=True)

# Guardar modelo
model.save("models/modelo_cnn.h5")

# Guardar LabelEncoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("Modelo y encoder guardados.")
```

Después, mueve los archivos generados a la carpeta `models/` del proyecto.

### 4. Verificar que los archivos están bien

```bash
python guardar_modelo.py
```

Deberías ver algo como:

```
Verificando archivos del modelo:
  ✓  models/modelo_cnn.h5
  ✓  models/label_encoder.pkl

Cargando para verificar...
  Modelo: OK — input (None, 64, 64, 1), output (None, 500)
  LabelEncoder: OK — 500 clases
  Primeras 10 clases: ['abandonar', 'abrir', ...]

✓ Todo listo. Puedes ejecutar main.py
```

### 5. Ejecutar la aplicación

```bash
python main.py
```

---

## Uso de la aplicación

1. **Iniciar** → el sistema abre el micrófono y empieza a escuchar.
2. **Habla** → el indicador cambia a **HABLANDO** cuando detecta voz.
3. **La palabra se muestra** en el panel inferior y se agrega a la transcripción.
4. **Ajustes VAD** (lado derecho):
   - *Umbral de energía*: sube si hay mucho ruido de fondo.
   - *Silencio fin de palabra*: sube si el sistema corta palabras largas.
5. **Detener** → cierra el micrófono.
6. **Limpiar** → borra la transcripción.

---

## Ajuste del umbral VAD

| Situación | Ajuste recomendado |
|-----------|-------------------|
| Mucho ruido de fondo | Aumentar umbral de energía (0.03–0.06) |
| Micrófono lejano / voz suave | Reducir umbral (0.005–0.015) |
| Corta palabras largas | Aumentar silencio fin de palabra (0.8–1.2 s) |
| Junta palabras distintas | Reducir silencio fin de palabra (0.4–0.5 s) |

Todos estos valores se pueden ajustar en `utils/config.py` o en tiempo
real desde los sliders de la interfaz.

---

## Parámetros del modelo (no modificar)

Estos valores deben coincidir **exactamente** con el entrenamiento:

| Parámetro | Valor |
|-----------|-------|
| Sample rate | 16 000 Hz |
| Duración | 1.0 s |
| N Mels | 64 |
| Time steps | 64 |
| N FFT | 1024 |
| Hop length | 256 |
| Win length | 512 |

Se definen en `utils/config.py` y se usan en `model/preprocessor.py`.

---

## Solución de problemas

**"Modelo no encontrado"**
→ Verifica que `models/modelo_cnn.h5` y `models/label_encoder.pkl` existen.

**"Error al abrir el micrófono"**
→ Verifica que el micrófono está conectado y que la aplicación tiene permisos.
→ En Windows: Configuración → Privacidad → Micrófono → permitir apps de escritorio.

**La palabra siempre es "???"**
→ La confianza es baja. Baja el umbral en `utils/config.py`: `UMBRAL_CONFIANZA = 0.2`.

**El sistema nunca detecta voz**
→ El umbral de energía es muy alto. Baja el slider de energía en la interfaz.

**El sistema detecta voz constantemente (ruido)**
→ El umbral es muy bajo. Sube el slider de energía.

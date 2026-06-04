"""
guardar_modelo.py
=================
Script auxiliar para guardar el modelo entrenado y el LabelEncoder
en el formato que espera la aplicación.

INSTRUCCIONES:
  1. Ejecuta primero tu script de entrenamiento (el que contiene el
     código del modelo CNN con TensorFlow).
  2. Al final del entrenamiento, ejecuta ESTE script en el mismo
     entorno Python donde entrenaste.
  3. Los archivos se guardarán en models/modelo_cnn.h5 y
     models/label_encoder.pkl listos para la aplicación.

NOTA: Este script asume que las variables 'model' y 'encoder'
existen en el mismo proceso. Adapta las rutas si tu script
guarda el modelo en otro lugar.

ALTERNATIVA — Si ya tienes el modelo guardado en otro formato:

  # Opción A: Desde un modelo ya guardado como SavedModel
  import tensorflow as tf
  modelo = tf.keras.models.load_model("ruta/a/mi/savedmodel")
  modelo.save("models/modelo_cnn.h5")

  # Opción B: Si el encoder está en memoria después del entrenamiento
  import pickle
  with open("models/label_encoder.pkl", "wb") as f:
      pickle.dump(encoder, f)
"""

import os
import pickle

# Crear carpeta si no existe
os.makedirs("models", exist_ok=True)

# ------------------------------------------------------------------
# BLOQUE A EJECUTAR AL FINAL DE TU SCRIPT DE ENTRENAMIENTO
# Pega estas líneas al final de tu script de entrenamiento, DESPUÉS
# de que 'model' y 'encoder' estén definidos.
# ------------------------------------------------------------------

# Guardar modelo Keras en formato H5
# model.save("models/modelo_cnn.h5")
# print("Modelo guardado en models/modelo_cnn.h5")

# Guardar LabelEncoder con pickle
# with open("models/label_encoder.pkl", "wb") as f:
#     pickle.dump(encoder, f)
# print("LabelEncoder guardado en models/label_encoder.pkl")

# ------------------------------------------------------------------
# VERIFICACIÓN — ejecuta esto para comprobar que los archivos están
# ------------------------------------------------------------------

if __name__ == "__main__":
    archivos = [
        "models/modelo_cnn.h5",
        "models/label_encoder.pkl",
    ]

    print("\nVerificando archivos del modelo:")
    todos_ok = True
    for ruta in archivos:
        existe = os.path.exists(ruta)
        estado = "✓" if existe else "✗ FALTA"
        print(f"  {estado}  {ruta}")
        if not existe:
            todos_ok = False

    if todos_ok:
        # Verificar que se pueden cargar
        import tensorflow as tf
        import pickle

        print("\nCargando para verificar...")
        modelo = tf.keras.models.load_model("models/modelo_cnn.h5")
        print(f"  Modelo: OK — input {modelo.input_shape}, output {modelo.output_shape}")

        with open("models/label_encoder.pkl", "rb") as f:
            enc = pickle.load(f)
        print(f"  LabelEncoder: OK — {len(enc.classes_)} clases")
        print(f"  Primeras 10 clases: {list(enc.classes_[:10])}")
        print("\n✓ Todo listo. Puedes ejecutar main.py")
    else:
        print("\n✗ Faltan archivos. Lee las instrucciones en este script.")
        print("  Pega el bloque de guardado al final de tu entrenamiento.")

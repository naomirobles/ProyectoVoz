"""
model/inference.py
==================
Carga del modelo CNN entrenado y ejecución de inferencia.

Responsabilidades:
  - Cargar el archivo .h5 / SavedModel de Keras.
  - Cargar el LabelEncoder (pickle) para convertir índices → palabras.
  - Recibir tensores preprocesados y retornar la palabra predicha
    junto con su probabilidad de confianza.
  - Exponer un umbral de confianza configurable: si la probabilidad
    máxima es menor al umbral, retorna "???" en lugar de una predicción
    poco confiable.

Archivos esperados (configurables en utils/config.py):
  - models/modelo_cnn.h5      (o carpeta SavedModel)
  - models/label_encoder.pkl
"""

import os
import pickle
import numpy as np

# TensorFlow se importa dentro de los métodos para que el import
# del módulo no falle si TF aún no está instalado en el entorno
# de desarrollo. En producción siempre estará disponible.
try:
    import tensorflow as tf
    TF_DISPONIBLE = True
except ImportError:
    TF_DISPONIBLE = False
    print("[Inference] ADVERTENCIA: TensorFlow no está instalado.")


class ModeloReconocedor:
    """
    Wrapper alrededor del modelo Keras CNN para reconocimiento de palabras.

    Parámetros
    ----------
    ruta_modelo : str
        Ruta al archivo del modelo (.h5 o carpeta SavedModel).
    ruta_encoder : str
        Ruta al archivo pickle del LabelEncoder.
    umbral_confianza : float
        Probabilidad mínima para aceptar una predicción.
        Si max(softmax) < umbral → retorna "???".
    """

    def __init__(
        self,
        ruta_modelo: str,
        ruta_encoder: str,
        umbral_confianza: float = 0.4,
    ):
        self.ruta_modelo = ruta_modelo
        self.ruta_encoder = ruta_encoder
        self.umbral_confianza = umbral_confianza

        self._modelo = None
        self._encoder = None
        self._cargado = False

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------

    def cargar(self) -> tuple[bool, str]:
        """
        Carga el modelo y el encoder desde disco.

        Retorna
        -------
        (éxito: bool, mensaje: str)
        """
        if not TF_DISPONIBLE:
            return False, "TensorFlow no está instalado."

        # Verificar que los archivos existen
        if not os.path.exists(self.ruta_modelo):
            return False, f"Modelo no encontrado: {self.ruta_modelo}"

        if not os.path.exists(self.ruta_encoder):
            return False, f"LabelEncoder no encontrado: {self.ruta_encoder}"

        try:
            # Cargar modelo Keras
            self._modelo = tf.keras.models.load_model(self.ruta_modelo)
            print(f"[Inference] Modelo cargado: {self.ruta_modelo}")
            print(f"[Inference] Input shape: {self._modelo.input_shape}")

        except Exception as e:
            return False, f"Error cargando modelo: {e}"

        try:
            # Cargar LabelEncoder
            with open(self.ruta_encoder, "rb") as f:
                self._encoder = pickle.load(f)
            print(f"[Inference] LabelEncoder cargado: {len(self._encoder.classes_)} clases.")

        except Exception as e:
            return False, f"Error cargando LabelEncoder: {e}"

        self._cargado = True
        return True, "Modelo cargado correctamente."

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def predecir(self, tensor: np.ndarray) -> tuple[str, float]:
        """
        Ejecuta inferencia sobre un tensor preprocesado.

        Parámetros
        ----------
        tensor : np.ndarray
            Shape (1, 64, 64, 1) — salida de AudioPreprocessor.process().

        Retorna
        -------
        (palabra: str, confianza: float)
            palabra     → texto de la clase predicha (o "???" si baja confianza).
            confianza   → probabilidad softmax de la clase ganadora [0.0, 1.0].
        """
        if not self._cargado:
            return "modelo_no_cargado", 0.0

        try:
            # Inferencia (silenciar el output de TF)
            probs = self._modelo.predict(tensor, verbose=0)  # shape (1, num_clases)
            probs_1d = probs[0]  # shape (num_clases,)

            indice = int(np.argmax(probs_1d))
            confianza = float(probs_1d[indice])

            # Umbral de confianza
            if confianza < self.umbral_confianza:
                return "???", confianza

            # Decodificar índice → texto
            palabra = self._encoder.inverse_transform([indice])[0]
            return str(palabra), confianza

        except Exception as e:
            print(f"[Inference] Error en predicción: {e}")
            return "error", 0.0

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def cargado(self) -> bool:
        return self._cargado

    @property
    def num_clases(self) -> int:
        if self._encoder is not None:
            return len(self._encoder.classes_)
        return 0

    @property
    def clases(self) -> list[str]:
        if self._encoder is not None:
            return list(self._encoder.classes_)
        return []

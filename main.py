"""
main.py
=======
Punto de entrada de la aplicación de reconocimiento de palabras.

Secuencia de arranque:
  1. Inicializar la aplicación Qt.
  2. Cargar el modelo CNN y el LabelEncoder desde disco.
  3. Mostrar la ventana principal.
  4. Entrar al bucle de eventos Qt.

Si el modelo no se encuentra, la aplicación se abre de todas formas
con los controles deshabilitados y un mensaje de aviso al usuario.
"""

import sys
import os

# ---- Silenciar mensajes innecesarios de TensorFlow ----
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      # Ocultar INFO y WARNING de TF
os.environ["PYTHONWARNINGS"] = "ignore"

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

from model.inference import ModeloReconocedor
from ui.main_window import VentanaPrincipal
from utils import config


def main() -> None:
    """Función principal: inicializa Qt, carga el modelo y muestra la UI."""

    # ------------------------------------------------------------------
    # 1. Crear la aplicación Qt
    # ------------------------------------------------------------------
    app = QApplication(sys.argv)
    app.setApplicationName(config.APP_TITLE)
    app.setOrganizationName("ESCOM-IPN")

    # Paleta oscura a nivel de aplicación (fallback si los QSS no aplican)
    app.setStyle("Fusion")

    # ------------------------------------------------------------------
    # 2. Cargar el modelo
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  RECONOCIMIENTO DE PALABRAS EN ESPAÑOL")
    print("  CNN + Espectrogramas Mel · Tiempo Real")
    print("=" * 60)

    # Asegurar que existe la carpeta models/
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    modelo = ModeloReconocedor(
        ruta_modelo=config.RUTA_MODELO,
        ruta_encoder=config.RUTA_ENCODER,
        umbral_confianza=config.UMBRAL_CONFIANZA,
    )

    exito, mensaje = modelo.cargar()
    if exito:
        print(f"[main] {mensaje}")
        print(f"[main] Clases disponibles ({modelo.num_clases}): {modelo.clases[:10]}...")
    else:
        print(f"[main] ADVERTENCIA — {mensaje}")
        print("[main] La aplicación se abrirá sin modelo cargado.")

    # ------------------------------------------------------------------
    # 3. Crear y mostrar la ventana principal
    # ------------------------------------------------------------------
    ventana = VentanaPrincipal(modelo)
    ventana.show()

    print("[main] Aplicación iniciada. Esperando eventos Qt...")

    # ------------------------------------------------------------------
    # 4. Bucle de eventos Qt
    # ------------------------------------------------------------------
    codigo_salida = app.exec()
    print(f"[main] Aplicación cerrada. Código de salida: {codigo_salida}")
    sys.exit(codigo_salida)


if __name__ == "__main__":
    main()

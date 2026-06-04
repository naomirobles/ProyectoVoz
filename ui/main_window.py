"""
ui/main_window.py
=================
Ventana principal de la aplicación de reconocimiento de palabras.

Diseño: tema oscuro industrial con acento rojo carmín.
Tipografía: JetBrains Mono (monoespaciada técnica) + Segoe UI / system font.

Componentes:
  - Indicador de estado animado (círculo de color + texto).
  - Área de transcripción acumulada (QTextEdit de solo lectura).
  - Panel de última palabra con confianza (barra de progreso).
  - Controles: Iniciar / Detener / Limpiar.
  - Panel de ajustes: umbral de energía, duración de silencio.
  - Barra de estado inferior con info del modelo.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QProgressBar,
    QSlider, QGroupBox, QStatusBar, QFrame, QSizePolicy,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QFont, QColor, QPalette, QTextCursor

from model.inference import ModeloReconocedor
from ui.worker import AudioWorker
from utils import config


class VentanaPrincipal(QMainWindow):
    """
    Ventana principal de la aplicación.

    Parámetros
    ----------
    modelo : ModeloReconocedor
        Instancia cargada del modelo CNN. Si no está cargado,
        se mostrará un aviso y los controles estarán deshabilitados.
    """

    def __init__(self, modelo: ModeloReconocedor):
        super().__init__()
        self.modelo = modelo
        self.worker: AudioWorker | None = None
        self._activo = False

        # Timer para el pulso del indicador de estado
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._animar_indicador)
        self._pulse_alpha = 255
        self._pulse_direction = -1  # -1 = decrecer, +1 = crecer
        self._estado_actual = "detenido"

        self._configurar_ventana()
        self._construir_ui()
        self._aplicar_estilos()
        self._conectar_slots()

        # Estado inicial
        self._actualizar_estado("detenido")

        if not modelo.cargado:
            self._mostrar_aviso_modelo()

    # ------------------------------------------------------------------
    # Configuración inicial
    # ------------------------------------------------------------------

    def _configurar_ventana(self) -> None:
        self.setWindowTitle(config.APP_TITLE)
        self.setMinimumSize(config.APP_WIDTH, config.APP_HEIGHT)
        self.resize(config.APP_WIDTH, config.APP_HEIGHT)

    # ------------------------------------------------------------------
    # Construcción de la UI
    # ------------------------------------------------------------------

    def _construir_ui(self) -> None:
        """Construye todos los widgets de la ventana."""
        central = QWidget()
        self.setCentralWidget(central)
        layout_raiz = QVBoxLayout(central)
        layout_raiz.setContentsMargins(20, 20, 20, 20)
        layout_raiz.setSpacing(16)

        # ---- Cabecera ----
        layout_raiz.addWidget(self._crear_cabecera())

        # ---- Indicador de estado ----
        layout_raiz.addWidget(self._crear_panel_estado())

        # ---- Última palabra detectada ----
        layout_raiz.addWidget(self._crear_panel_ultima_palabra())

        # ---- Transcripción ----
        layout_raiz.addWidget(self._crear_panel_transcripcion(), stretch=1)

        # ---- Controles y ajustes ----
        fila_inferior = QHBoxLayout()
        fila_inferior.setSpacing(16)
        fila_inferior.addWidget(self._crear_panel_controles())
        fila_inferior.addWidget(self._crear_panel_ajustes())
        layout_raiz.addLayout(fila_inferior)

        # ---- Barra de estado ----
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._actualizar_status_bar()

    def _crear_cabecera(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Línea decorativa izquierda
        linea = QFrame()
        linea.setFixedSize(4, 36)
        linea.setStyleSheet(f"background: {config.COLOR_ACENTO}; border-radius: 2px;")
        layout.addWidget(linea)

        layout.addSpacing(12)

        # Títulos
        col_titulos = QVBoxLayout()
        col_titulos.setSpacing(2)

        titulo = QLabel("RECONOCIMIENTO DE VOZ")
        titulo.setObjectName("titulo_app")
        col_titulos.addWidget(titulo)

        subtitulo = QLabel("Español · CNN + Espectrogramas Mel · Tiempo Real")
        subtitulo.setObjectName("subtitulo_app")
        col_titulos.addWidget(subtitulo)

        layout.addLayout(col_titulos)
        layout.addStretch()

        # Chip: número de clases
        if self.modelo.cargado:
            chip = QLabel(f"  {self.modelo.num_clases} clases  ")
            chip.setObjectName("chip_clases")
            layout.addWidget(chip)

        return widget

    def _crear_panel_estado(self) -> QWidget:
        widget = QWidget()
        widget.setObjectName("panel_estado")
        widget.setFixedHeight(80)

        layout = QHBoxLayout(widget)
        layout.setContentsMargins(20, 0, 20, 0)

        # Indicador circular (LED)
        self.lbl_led = QLabel("●")
        self.lbl_led.setObjectName("led")
        self.lbl_led.setFixedWidth(40)
        layout.addWidget(self.lbl_led)

        layout.addSpacing(12)

        # Texto de estado
        col = QVBoxLayout()
        col.setSpacing(2)

        self.lbl_estado = QLabel("DETENIDO")
        self.lbl_estado.setObjectName("lbl_estado")
        col.addWidget(self.lbl_estado)

        self.lbl_estado_desc = QLabel("Presiona Iniciar para comenzar la escucha")
        self.lbl_estado_desc.setObjectName("lbl_estado_desc")
        col.addWidget(self.lbl_estado_desc)

        layout.addLayout(col)
        layout.addStretch()

        # Contador de palabras
        col_contador = QVBoxLayout()
        col_contador.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.lbl_contador_num = QLabel("0")
        self.lbl_contador_num.setObjectName("contador_num")
        self.lbl_contador_num.setAlignment(Qt.AlignmentFlag.AlignRight)
        col_contador.addWidget(self.lbl_contador_num)

        lbl_palabras = QLabel("palabras")
        lbl_palabras.setObjectName("contador_label")
        lbl_palabras.setAlignment(Qt.AlignmentFlag.AlignRight)
        col_contador.addWidget(lbl_palabras)

        layout.addLayout(col_contador)

        return widget

    def _crear_panel_ultima_palabra(self) -> QWidget:
        widget = QWidget()
        widget.setObjectName("panel_ultima")
        widget.setFixedHeight(70)

        layout = QHBoxLayout(widget)
        layout.setContentsMargins(20, 10, 20, 10)

        lbl_etiqueta = QLabel("ÚLTIMA PALABRA:")
        lbl_etiqueta.setObjectName("etiqueta_campo")
        layout.addWidget(lbl_etiqueta)

        layout.addSpacing(12)

        self.lbl_ultima_palabra = QLabel("—")
        self.lbl_ultima_palabra.setObjectName("ultima_palabra")
        layout.addWidget(self.lbl_ultima_palabra)

        layout.addStretch()

        # Barra de confianza
        col_confianza = QVBoxLayout()
        col_confianza.setSpacing(4)

        lbl_conf_etq = QLabel("Confianza")
        lbl_conf_etq.setObjectName("etiqueta_campo")
        col_confianza.addWidget(lbl_conf_etq)

        self.barra_confianza = QProgressBar()
        self.barra_confianza.setRange(0, 100)
        self.barra_confianza.setValue(0)
        self.barra_confianza.setFixedWidth(200)
        self.barra_confianza.setFixedHeight(12)
        self.barra_confianza.setTextVisible(False)
        col_confianza.addWidget(self.barra_confianza)

        self.lbl_confianza_val = QLabel("0 %")
        self.lbl_confianza_val.setObjectName("etiqueta_campo")
        col_confianza.addWidget(self.lbl_confianza_val)

        layout.addLayout(col_confianza)

        return widget

    def _crear_panel_transcripcion(self) -> QGroupBox:
        grupo = QGroupBox("TRANSCRIPCIÓN")
        layout = QVBoxLayout(grupo)

        self.txt_transcripcion = QTextEdit()
        self.txt_transcripcion.setReadOnly(True)
        self.txt_transcripcion.setObjectName("txt_transcripcion")
        self.txt_transcripcion.setPlaceholderText(
            "Las palabras reconocidas aparecerán aquí..."
        )
        layout.addWidget(self.txt_transcripcion)

        return grupo

    def _crear_panel_controles(self) -> QGroupBox:
        grupo = QGroupBox("CONTROLES")
        layout = QVBoxLayout(grupo)
        layout.setSpacing(10)

        self.btn_iniciar = QPushButton("▶  INICIAR")
        self.btn_iniciar.setObjectName("btn_iniciar")
        self.btn_iniciar.setFixedHeight(44)
        self.btn_iniciar.setEnabled(self.modelo.cargado)
        layout.addWidget(self.btn_iniciar)

        self.btn_detener = QPushButton("■  DETENER")
        self.btn_detener.setObjectName("btn_detener")
        self.btn_detener.setFixedHeight(44)
        self.btn_detener.setEnabled(False)
        layout.addWidget(self.btn_detener)

        self.btn_limpiar = QPushButton("↺  LIMPIAR")
        self.btn_limpiar.setObjectName("btn_limpiar")
        self.btn_limpiar.setFixedHeight(44)
        layout.addWidget(self.btn_limpiar)

        layout.addStretch()
        return grupo

    def _crear_panel_ajustes(self) -> QGroupBox:
        grupo = QGroupBox("AJUSTES VAD")
        layout = QVBoxLayout(grupo)
        layout.setSpacing(12)

        # --- Umbral de energía ---
        lbl_thresh = QLabel(f"Umbral de energía: {config.VAD_ENERGY_THRESHOLD:.3f}")
        lbl_thresh.setObjectName("lbl_ajuste")
        self._lbl_thresh = lbl_thresh
        layout.addWidget(lbl_thresh)

        self.slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh.setRange(1, 100)    # valores × 0.001
        self.slider_thresh.setValue(int(config.VAD_ENERGY_THRESHOLD * 1000))
        layout.addWidget(self.slider_thresh)

        hint_thresh = QLabel("↑ Sube si hay mucho ruido de fondo")
        hint_thresh.setObjectName("lbl_hint")
        layout.addWidget(hint_thresh)

        # --- Duración de silencio ---
        lbl_sil = QLabel(f"Silencio fin de palabra: {config.VAD_SILENCE_DURATION_S:.2f} s")
        lbl_sil.setObjectName("lbl_ajuste")
        self._lbl_sil = lbl_sil
        layout.addWidget(lbl_sil)

        self.slider_sil = QSlider(Qt.Orientation.Horizontal)
        self.slider_sil.setRange(20, 150)      # valores × 0.01 segundos
        self.slider_sil.setValue(int(config.VAD_SILENCE_DURATION_S * 100))
        layout.addWidget(self.slider_sil)

        hint_sil = QLabel("↑ Sube si corta palabras largas")
        hint_sil.setObjectName("lbl_hint")
        layout.addWidget(hint_sil)

        layout.addStretch()
        return grupo

    # ------------------------------------------------------------------
    # Estilos
    # ------------------------------------------------------------------

    def _aplicar_estilos(self) -> None:
        self.setStyleSheet(f"""
            /* ---- Base ---- */
            QMainWindow, QWidget {{
                background-color: {config.COLOR_FONDO};
                color: {config.COLOR_TEXTO};
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                font-size: 13px;
            }}

            /* ---- Cabecera ---- */
            #titulo_app {{
                font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
                font-size: 18px;
                font-weight: 700;
                color: {config.COLOR_TEXTO};
                letter-spacing: 3px;
            }}
            #subtitulo_app {{
                font-size: 11px;
                color: {config.COLOR_TEXTO_GRIS};
                letter-spacing: 1px;
            }}
            #chip_clases {{
                background: {config.COLOR_ACENTO};
                color: white;
                border-radius: 10px;
                font-size: 11px;
                font-weight: 600;
                padding: 4px 8px;
            }}

            /* ---- Panel estado ---- */
            #panel_estado {{
                background: {config.COLOR_PANEL};
                border-radius: 10px;
                border: 1px solid #2a2a2a;
            }}
            #led {{
                font-size: 28px;
                color: {config.COLOR_IDLE};
            }}
            #lbl_estado {{
                font-family: 'JetBrains Mono', 'Consolas', monospace;
                font-size: 16px;
                font-weight: 700;
                letter-spacing: 2px;
                color: {config.COLOR_IDLE};
            }}
            #lbl_estado_desc {{
                font-size: 11px;
                color: {config.COLOR_TEXTO_GRIS};
            }}
            #contador_num {{
                font-family: 'JetBrains Mono', 'Consolas', monospace;
                font-size: 32px;
                font-weight: 700;
                color: {config.COLOR_ACENTO};
            }}
            #contador_label {{
                font-size: 10px;
                color: {config.COLOR_TEXTO_GRIS};
                text-transform: uppercase;
                letter-spacing: 1px;
            }}

            /* ---- Panel última palabra ---- */
            #panel_ultima {{
                background: {config.COLOR_PANEL};
                border-radius: 10px;
                border: 1px solid #2a2a2a;
            }}
            #etiqueta_campo {{
                font-size: 10px;
                color: {config.COLOR_TEXTO_GRIS};
                letter-spacing: 1px;
                text-transform: uppercase;
            }}
            #ultima_palabra {{
                font-family: 'JetBrains Mono', 'Consolas', monospace;
                font-size: 22px;
                font-weight: 700;
                color: {config.COLOR_TEXTO};
            }}

            /* ---- Barra de confianza ---- */
            QProgressBar {{
                background: #2a2a2a;
                border-radius: 6px;
                border: none;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {config.COLOR_ACENTO_SUAVE}, stop:1 {config.COLOR_ACENTO});
                border-radius: 6px;
            }}

            /* ---- Transcripción ---- */
            QGroupBox {{
                font-family: 'JetBrains Mono', 'Consolas', monospace;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 2px;
                color: {config.COLOR_TEXTO_GRIS};
                border: 1px solid #2a2a2a;
                border-radius: 10px;
                margin-top: 14px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                left: 16px;
            }}
            #txt_transcripcion {{
                background: #111111;
                border: none;
                border-radius: 8px;
                color: {config.COLOR_TEXTO};
                font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
                font-size: 15px;
                line-height: 1.8;
                padding: 12px;
                selection-background-color: {config.COLOR_ACENTO};
            }}

            /* ---- Botones ---- */
            QPushButton {{
                border-radius: 8px;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 1px;
                border: none;
                padding: 0 16px;
            }}
            #btn_iniciar {{
                background: {config.COLOR_ACENTO};
                color: white;
            }}
            #btn_iniciar:hover {{
                background: #e74c3c;
            }}
            #btn_iniciar:disabled {{
                background: #3a3a3a;
                color: {config.COLOR_TEXTO_GRIS};
            }}
            #btn_detener {{
                background: #2a2a2a;
                color: {config.COLOR_TEXTO};
                border: 1px solid #3a3a3a;
            }}
            #btn_detener:hover {{
                background: #333;
            }}
            #btn_detener:disabled {{
                color: {config.COLOR_TEXTO_GRIS};
            }}
            #btn_limpiar {{
                background: transparent;
                color: {config.COLOR_TEXTO_GRIS};
                border: 1px solid #2a2a2a;
            }}
            #btn_limpiar:hover {{
                color: {config.COLOR_TEXTO};
                border-color: #444;
            }}

            /* ---- Sliders ---- */
            QSlider::groove:horizontal {{
                height: 4px;
                background: #2a2a2a;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                width: 16px;
                height: 16px;
                background: {config.COLOR_ACENTO};
                border-radius: 8px;
                margin: -6px 0;
            }}
            QSlider::sub-page:horizontal {{
                background: {config.COLOR_ACENTO};
                border-radius: 2px;
            }}

            /* ---- Labels de ajuste ---- */
            #lbl_ajuste {{
                font-size: 11px;
                font-weight: 600;
                color: {config.COLOR_TEXTO};
            }}
            #lbl_hint {{
                font-size: 10px;
                color: {config.COLOR_TEXTO_GRIS};
            }}

            /* ---- Status bar ---- */
            QStatusBar {{
                background: #111;
                color: {config.COLOR_TEXTO_GRIS};
                font-size: 10px;
                border-top: 1px solid #2a2a2a;
            }}
        """)

    # ------------------------------------------------------------------
    # Conexión de señales/slots
    # ------------------------------------------------------------------

    def _conectar_slots(self) -> None:
        self.btn_iniciar.clicked.connect(self._on_iniciar)
        self.btn_detener.clicked.connect(self._on_detener)
        self.btn_limpiar.clicked.connect(self._on_limpiar)
        self.slider_thresh.valueChanged.connect(self._on_thresh_changed)
        self.slider_sil.valueChanged.connect(self._on_sil_changed)

    # ------------------------------------------------------------------
    # Slots de botones
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_iniciar(self) -> None:
        if self._activo:
            return

        # Crear worker y conectar sus señales
        self.worker = AudioWorker(self.modelo)
        self.worker.estado_cambiado.connect(self._on_estado_cambiado)
        self.worker.palabra_detectada.connect(self._on_palabra_detectada)
        self.worker.error_ocurrido.connect(self._on_error)

        self.worker.iniciar()
        self._activo = True

        self.btn_iniciar.setEnabled(False)
        self.btn_detener.setEnabled(True)

        # Aplicar ajustes actuales de sliders al worker
        self.worker.set_energy_threshold(self.slider_thresh.value() / 1000.0)
        self.worker.set_silence_duration(self.slider_sil.value() / 100.0)

    @pyqtSlot()
    def _on_detener(self) -> None:
        if not self._activo or self.worker is None:
            return

        self.worker.detener()
        self.worker = None
        self._activo = False

        self.btn_iniciar.setEnabled(self.modelo.cargado)
        self.btn_detener.setEnabled(False)
        self._actualizar_estado("detenido")

    @pyqtSlot()
    def _on_limpiar(self) -> None:
        self.txt_transcripcion.clear()
        self.lbl_ultima_palabra.setText("—")
        self.barra_confianza.setValue(0)
        self.lbl_confianza_val.setText("0 %")
        self.lbl_contador_num.setText("0")
        self._contador_palabras = 0

    # ------------------------------------------------------------------
    # Slots del worker
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def _on_estado_cambiado(self, estado: str) -> None:
        self._actualizar_estado(estado)

    @pyqtSlot(str, float)
    def _on_palabra_detectada(self, palabra: str, confianza: float) -> None:
        """Actualiza la UI con la nueva palabra reconocida."""
        # Última palabra
        self.lbl_ultima_palabra.setText(palabra)

        # Barra de confianza
        pct = int(confianza * 100)
        self.barra_confianza.setValue(pct)
        self.lbl_confianza_val.setText(f"{pct} %")

        # Agregar a transcripción
        self.txt_transcripcion.insertPlainText(f"{palabra} ")
        # Hacer scroll al final
        cursor = self.txt_transcripcion.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.txt_transcripcion.setTextCursor(cursor)

        # Contador
        contador = int(self.lbl_contador_num.text()) + 1
        self.lbl_contador_num.setText(str(contador))

    @pyqtSlot(str)
    def _on_error(self, mensaje: str) -> None:
        QMessageBox.critical(self, "Error", mensaje)
        self._on_detener()

    # ------------------------------------------------------------------
    # Slots de ajustes
    # ------------------------------------------------------------------

    @pyqtSlot(int)
    def _on_thresh_changed(self, valor: int) -> None:
        v = valor / 1000.0
        self._lbl_thresh.setText(f"Umbral de energía: {v:.3f}")
        if self.worker:
            self.worker.set_energy_threshold(v)

    @pyqtSlot(int)
    def _on_sil_changed(self, valor: int) -> None:
        v = valor / 100.0
        self._lbl_sil.setText(f"Silencio fin de palabra: {v:.2f} s")
        if self.worker:
            self.worker.set_silence_duration(v)

    # ------------------------------------------------------------------
    # Actualización de estado / animación
    # ------------------------------------------------------------------

    def _actualizar_estado(self, estado: str) -> None:
        self._estado_actual = estado
        mapa = {
            "escuchando":  (config.COLOR_LISTENING,   "ESCUCHANDO",  "Esperando actividad de voz..."),
            "hablando":    (config.COLOR_SPEAKING,    "HABLANDO",    "Grabando segmento de voz..."),
            "procesando":  (config.COLOR_PROCESSING,  "PROCESANDO",  "Ejecutando inferencia CNN..."),
            "detenido":    (config.COLOR_IDLE,         "DETENIDO",    "Presiona Iniciar para comenzar la escucha"),
        }
        color, texto, desc = mapa.get(estado, (config.COLOR_IDLE, estado.upper(), ""))

        self.lbl_led.setStyleSheet(f"color: {color}; font-size: 28px;")
        self.lbl_estado.setStyleSheet(
            f"font-family: 'JetBrains Mono', 'Consolas', monospace; "
            f"font-size: 16px; font-weight: 700; letter-spacing: 2px; color: {color};"
        )
        self.lbl_estado.setText(texto)
        self.lbl_estado_desc.setText(desc)

        # Activar / desactivar pulso
        if estado in ("hablando", "escuchando"):
            if not self._pulse_timer.isActive():
                self._pulse_timer.start(50)  # actualizar cada 50 ms
        else:
            self._pulse_timer.stop()
            self.lbl_led.setStyleSheet(f"color: {color}; font-size: 28px;")

    def _animar_indicador(self) -> None:
        """Animación de pulso: hace que el LED oscile en opacidad."""
        self._pulse_alpha += self._pulse_direction * 12
        if self._pulse_alpha <= 80:
            self._pulse_alpha = 80
            self._pulse_direction = 1
        elif self._pulse_alpha >= 255:
            self._pulse_alpha = 255
            self._pulse_direction = -1

        colores = {
            "escuchando": config.COLOR_LISTENING,
            "hablando":   config.COLOR_SPEAKING,
        }
        color_hex = colores.get(self._estado_actual, config.COLOR_IDLE)
        # Aplicar opacidad a través de rgba aproximado
        qc = QColor(color_hex)
        qc.setAlpha(self._pulse_alpha)
        rgba = f"rgba({qc.red()},{qc.green()},{qc.blue()},{self._pulse_alpha})"
        self.lbl_led.setStyleSheet(f"color: {rgba}; font-size: 28px;")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mostrar_aviso_modelo(self) -> None:
        QMessageBox.warning(
            self,
            "Modelo no encontrado",
            f"No se encontró el modelo en:\n{config.RUTA_MODELO}\n\n"
            f"Coloca los archivos en la carpeta 'models/' y reinicia la app.\n\n"
            f"Archivos necesarios:\n"
            f"  • models/modelo_cnn.h5\n"
            f"  • models/label_encoder.pkl"
        )

    def _actualizar_status_bar(self) -> None:
        if self.modelo.cargado:
            msg = (f"✓ Modelo: {config.RUTA_MODELO}  |  "
                   f"{self.modelo.num_clases} clases  |  "
                   f"SR: {config.SAMPLE_RATE} Hz  |  "
                   f"Mel: {config.N_MELS}×{config.TIME_STEPS}")
        else:
            msg = "✗ Modelo no cargado. Coloca los archivos en models/ y reinicia."
        self._status_bar.showMessage(msg)

    # ------------------------------------------------------------------
    # Cierre
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Asegurar que el hilo de audio se detenga al cerrar."""
        if self._activo and self.worker:
            self.worker.detener()
        event.accept()

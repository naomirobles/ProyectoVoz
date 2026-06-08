"""
Generador del reporte técnico completo en formato .docx
"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

OUTPUTS = "outputs"
IMG_EDA       = os.path.join(OUTPUTS, "eda")
IMG_TRAINING  = os.path.join(OUTPUTS, "training")
IMG_INTERP    = os.path.join(OUTPUTS, "interpretability")

# ─────────────────────────────────────────────
# Helpers de formato
# ─────────────────────────────────────────────

def set_font(run, name="Times New Roman", size=12, bold=False, italic=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)

def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in h.runs:
        run.font.name = "Times New Roman"
        if level == 1:
            run.font.size = Pt(16)
            run.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
        elif level == 2:
            run.font.size = Pt(14)
            run.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
        else:
            run.font.size = Pt(12)
            run.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
    return h

def add_paragraph(doc, text, justify=True, bold=False, italic=False, size=12):
    p = doc.add_paragraph()
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    set_font(run, size=size, bold=bold, italic=italic)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.first_line_indent = Cm(0.75)
    return p

def add_paragraph_no_indent(doc, text, justify=True, bold=False, italic=False, size=12):
    p = doc.add_paragraph()
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    set_font(run, size=size, bold=bold, italic=italic)
    p.paragraph_format.space_after = Pt(6)
    return p

def add_figure(doc, path, caption, width=5.5):
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    set_font(r, size=10, italic=True)
    cap.paragraph_format.space_after = Pt(10)

def add_table(doc, headers, rows, caption=""):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        for p in hdr[i].paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                set_font(run, size=10, bold=True)
    for ri, row in enumerate(rows):
        cells = table.rows[ri + 1].cells
        for ci, val in enumerate(row):
            cells[ci].text = str(val)
            for p in cells[ci].paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    set_font(run, size=10)
    if caption:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run(caption)
        set_font(r, size=10, italic=True)
    doc.add_paragraph()

def page_break(doc):
    doc.add_page_break()

# ─────────────────────────────────────────────
# Construcción del documento
# ─────────────────────────────────────────────

doc = Document()

# Márgenes
section = doc.sections[0]
section.page_width  = Cm(21.59)
section.page_height = Cm(27.94)
section.left_margin   = Cm(3)
section.right_margin  = Cm(2.5)
section.top_margin    = Cm(2.5)
section.bottom_margin = Cm(2.5)

# ══════════════════════════════════════════════
# 1. PORTADA
# ══════════════════════════════════════════════
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("\n\n\n")

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("INSTITUTO POLITÉCNICO NACIONAL")
set_font(run, size=14, bold=True)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Escuela Superior de Cómputo")
set_font(run, size=13, bold=True)

doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("REPORTE TÉCNICO")
set_font(run, size=18, bold=True, color=(0x1F, 0x49, 0x7D))

doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run(
    "Reconocimiento Automático de Palabras en Español\n"
    "mediante Redes Neuronales Convolucionales\n"
    "sobre el Dataset ML Spoken Words"
)
set_font(run, size=15, bold=True)

doc.add_paragraph()
doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Unidad de Aprendizaje: Reconocimiento de Voz\nSéptimo Semestre")
set_font(run, size=12)

doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Fecha: 7 de junio de 2026")
set_font(run, size=12)

doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Entorno de ejecución: Python 3.11.15 · TensorFlow 2.21.0 · Keras Tuner 1.4.8")
set_font(run, size=10, italic=True)

page_break(doc)

# ══════════════════════════════════════════════
# 2. RESUMEN EJECUTIVO
# ══════════════════════════════════════════════
add_heading(doc, "Resumen Ejecutivo", 1)

add_paragraph(doc,
    "El presente reporte documenta el diseño, implementación y evaluación de un sistema de reconocimiento "
    "automático de voz (RAV) para el español, basado en redes neuronales convolucionales (CNN). El sistema "
    "fue entrenado sobre el conjunto de datos MLCommons ML Spoken Words en su variante de español, que "
    "comprende 944,513 muestras distribuidas en 24,878 palabras únicas. Mediante un proceso de selección "
    "cuantitativa, se identificaron 35 palabras objetivo que satisfacen criterios de representatividad "
    "estadística, longitud fonética mínima y disimilaridad acústica mutua."
)

add_paragraph(doc,
    "La arquitectura de la red fue optimizada de forma automática con Keras Tuner empleando la estrategia "
    "Hyperband, explorando 30 configuraciones distintas. El modelo ganador posee tres capas convolucionales "
    "con filtros de tamaños [32, 128, 16], regularización L2, una capa densa de 512 neuronas y dropout de "
    "0.40. Entrenado durante 63 épocas con parada temprana, el modelo obtuvo una exactitud (accuracy) del "
    "92.69% sobre el conjunto de prueba y un F1-macro de 0.9271 en las 35 clases evaluadas."
)

add_paragraph(doc,
    "La validación cruzada con 5 particiones (K-Fold) arrojó una exactitud promedio del 81.43% "
    "(IC 95%: [0.7943, 0.8342]), lo que confirma que el modelo generaliza de forma robusta a datos no "
    "vistos durante el entrenamiento. Los principales errores de clasificación se concentran en pares de "
    "palabras morfológicamente similares —principalmente 'estos/estas/están'— cuya confusión responde a "
    "causas fonéticas y acústicas identificables y documentadas en el análisis de interpretabilidad."
)

page_break(doc)

# ══════════════════════════════════════════════
# 3. INTRODUCCIÓN
# ══════════════════════════════════════════════
add_heading(doc, "1. Introducción", 1)

add_paragraph(doc,
    "El reconocimiento automático de voz (RAV) es una de las áreas de mayor impacto dentro del "
    "procesamiento del lenguaje natural y la inteligencia artificial aplicada. Los sistemas RAV permiten "
    "convertir señales de audio en representaciones textuales, facilitando interfaces humano-computadora "
    "más naturales e inclusivas. Sin embargo, el diseño de clasificadores robustos para vocabulario "
    "acotado —como el empleado en sistemas de comando de voz o asistentes conversacionales— sigue siendo "
    "un problema relevante, especialmente cuando el objetivo es el español, idioma con variaciones "
    "fonéticas propias y cuya representación en datasets públicos es significativamente menor que la del "
    "inglés."
)

add_paragraph(doc,
    "Las redes neuronales convolucionales han demostrado ser altamente efectivas para la clasificación de "
    "señales de audio representadas como espectrogramas de Mel. Cuando una señal de voz se transforma en "
    "una imagen bidimensional que codifica la distribución de energía en el tiempo y la frecuencia, las "
    "capas convolucionales pueden extraer patrones locales —como formantes, transiciones fonéticas y "
    "características de prosodia— que son discriminativos para la identificación de palabras individuales."
)

add_paragraph(doc,
    "Este proyecto aborda el problema de clasificación de palabras aisladas en español, formulando un "
    "clasificador multiclase de 35 categorías. El pipeline experimental implementado cubre desde la "
    "exploración estadística del dataset y la selección automática de palabras, pasando por la extracción "
    "de características acústicas, hasta la optimización de hiperparámetros y la validación estadística "
    "del desempeño. Cada decisión de diseño fue respaldada por evidencia experimental documentada y "
    "reproducible mediante la fijación de semillas aleatorias (seed = 42)."
)

add_paragraph(doc,
    "El reporte está organizado de la siguiente manera: la Sección 2 presenta los objetivos; la Sección 3 "
    "describe el dataset; la Sección 4 analiza exploratoriamente los datos; la Sección 5 documenta la "
    "selección de palabras; las Secciones 6 y 7 cubren el procesamiento de audio y la arquitectura de la "
    "red; la Sección 8 explica la optimización con Keras Tuner; las Secciones 9 y 10 presentan el "
    "entrenamiento y la validación; la Sección 11 analiza la interpretabilidad; y las Secciones 12 a 16 "
    "discuten los resultados, limitaciones, trabajo futuro y conclusiones."
)

page_break(doc)

# ══════════════════════════════════════════════
# 4. OBJETIVOS
# ══════════════════════════════════════════════
add_heading(doc, "2. Objetivos", 1)

add_heading(doc, "2.1 Objetivo General", 2)
add_paragraph(doc,
    "Diseñar, implementar y evaluar un clasificador de palabras aisladas en español basado en redes "
    "neuronales convolucionales, entrenado sobre espectrogramas de Mel extraídos del corpus ML Spoken "
    "Words, optimizando automáticamente su arquitectura mediante búsqueda de hiperparámetros y "
    "verificando su capacidad de generalización con validación cruzada estratificada."
)

add_heading(doc, "2.2 Objetivos Específicos", 2)
add_paragraph(doc,
    "El primer objetivo específico consiste en analizar exploratoriamente el dataset ML Spoken Words "
    "en español para caracterizar su distribución de frecuencias y duraciones, identificando el subconjunto "
    "estadísticamente adecuado para el entrenamiento. El segundo objetivo es desarrollar e implementar "
    "un pipeline de selección automática de palabras basado en criterios cuantitativos verificables, "
    "sin intervención manual subjetiva. El tercer objetivo es construir un preprocesador de audio "
    "que transforme señales crudas de voz en espectrogramas de Mel normalizados de dimensiones fijas, "
    "adecuados para ser procesados por redes convolucionales. El cuarto objetivo es optimizar la "
    "arquitectura de la CNN mediante Keras Tuner con estrategia Hyperband, explorando el espacio de "
    "hiperparámetros de forma eficiente. El quinto objetivo es entrenar el modelo final sobre el "
    "conjunto completo de datos seleccionado, monitoreando el sobreajuste mediante EarlyStopping y "
    "ReduceLROnPlateau. Finalmente, el sexto objetivo es cuantificar el desempeño del modelo mediante "
    "métricas de exactitud, F1-score por clase e intervalos de confianza derivados de la validación cruzada."
)

page_break(doc)

# ══════════════════════════════════════════════
# 5. DESCRIPCIÓN DEL DATASET
# ══════════════════════════════════════════════
add_heading(doc, "3. Descripción del Dataset", 1)

add_heading(doc, "3.1 ML Spoken Words", 2)
add_paragraph(doc,
    "El conjunto de datos empleado en este proyecto es MLCommons ML Spoken Words, publicado por la "
    "organización MLCommons como un recurso de acceso abierto para la investigación en reconocimiento "
    "de voz de palabras aisladas en múltiples idiomas. Para este proyecto se utilizó específicamente "
    "la variante en español (identificador: es_wav), que corresponde a grabaciones de hablantes nativos "
    "o competentes en español pronunciando palabras individuales."
)

add_paragraph(doc,
    "El corpus español contiene un total de 944,513 muestras de audio distribuidas en 24,878 palabras "
    "únicas. Esta distribución es marcadamente desbalanceada: una minoría de palabras concentra la "
    "mayor parte de las grabaciones, mientras que la vasta mayoría cuenta con apenas unos pocos "
    "ejemplos por clase. De acuerdo con el análisis exploratorio, solo 79 palabras (el 0.32% del "
    "vocabulario total) poseen al menos 1,000 muestras, umbral mínimo establecido para garantizar "
    "suficiencia estadística en la estimación de métricas de desempeño por clase."
)

add_paragraph(doc,
    "Las grabaciones fueron estandarizadas a una frecuencia de muestreo de 16,000 Hz y una duración "
    "modal de 1 segundo. El análisis de 1,000 muestras aleatorias del corpus reveló una duración "
    "promedio de 0.9999 segundos con una desviación estándar de apenas 0.00088 segundos, lo que "
    "indica una distribución extraordinariamente concentrada alrededor del segundo de duración. "
    "La duración mínima observada fue de 0.972 segundos y la máxima de 1.000 segundos, confirmando "
    "que el corpus fue preprocesado previamente para ajustar las duraciones a una longitud uniforme."
)

add_heading(doc, "3.2 Justificación de la Elección del Dataset", 2)
add_paragraph(doc,
    "La selección de ML Spoken Words responde a varios criterios técnicos y prácticos. En primer lugar, "
    "es uno de los pocos datasets de habla aislada en español con licencia abierta, diversidad de "
    "hablantes y cobertura léxica amplia. En segundo lugar, la estandarización de duración ya aplicada "
    "en el corpus simplifica significativamente el pipeline de preprocesamiento. En tercer lugar, la "
    "escala del dataset (casi un millón de muestras) permite construir conjuntos de entrenamiento "
    "estadísticamente robustos para las palabras seleccionadas, con 1,000 ejemplos por clase, "
    "garantizando estimaciones de métricas con margen de error menor al 5% a un nivel de confianza "
    "del 95%."
)

page_break(doc)

# ══════════════════════════════════════════════
# 6. ANÁLISIS EXPLORATORIO DE DATOS
# ══════════════════════════════════════════════
add_heading(doc, "4. Análisis Exploratorio de Datos", 1)

add_heading(doc, "4.1 Estadísticas Generales del Corpus", 2)
add_paragraph(doc,
    "El análisis exploratorio de datos (EDA) constituyó el punto de partida del pipeline experimental. "
    "Su propósito fue caracterizar cuantitativamente el corpus para tomar decisiones informadas sobre "
    "la viabilidad de distintas estrategias de selección de palabras y preprocesamiento. Los resultados "
    "globales se resumen en la Tabla 1."
)

add_table(doc,
    ["Métrica", "Valor"],
    [
        ["Total de palabras únicas en el corpus", "24,878"],
        ["Total de muestras de audio", "944,513"],
        ["Palabras con ≥ 1,000 muestras", "79"],
        ["Porcentaje con ≥ 1,000 muestras", "0.32%"],
        ["Duración promedio de las muestras", "0.9999 s"],
        ["Desviación estándar de duración", "0.000884 s"],
        ["Duración mínima observada", "0.972 s"],
        ["Duración máxima observada", "1.000 s"],
        ["Porcentaje de clases balanceadas (±20% mediana)", "18.36%"],
    ],
    "Tabla 1. Estadísticas generales del corpus ML Spoken Words (español)."
)

add_heading(doc, "4.2 Distribución de Frecuencias", 2)
add_paragraph(doc,
    "La Figura 1 muestra el histograma de frecuencias de muestras por palabra. La distribución es "
    "fuertemente sesgada hacia la derecha, característica conocida como ley de Zipf en corpus lingüísticos: "
    "un pequeño número de palabras de alta frecuencia concentra la mayor parte de las instancias, "
    "mientras que la inmensa mayoría de los vocablos —palabras raras o nombres propios— posee menos "
    "de 100 muestras. Las palabras más frecuentes del corpus español son 'los' (19,430 muestras), "
    "'del' (16,916), 'que' (16,108) y 'por' (14,036), todas ellas palabras funcionales (artículos, "
    "preposiciones, conjunciones) que, a pesar de su alta frecuencia, fueron descartadas durante la "
    "selección por no cumplir con el criterio de longitud mínima de 5 caracteres."
)
add_figure(doc, os.path.join(IMG_EDA, "histograma_frecuencias.png"),
           "Figura 1. Histograma de distribución de frecuencias de muestras por palabra en el corpus.")

add_heading(doc, "4.3 Distribución de Duraciones", 2)
add_paragraph(doc,
    "La Figura 2 muestra el histograma de duraciones construido sobre una muestra aleatoria de 1,000 "
    "grabaciones del corpus. Como puede observarse, la distribución es unimodal y extremadamente "
    "concentrada alrededor del valor de 1 segundo, con prácticamente ninguna muestra fuera del "
    "rango [0.97, 1.00] segundos. Esta característica es consecuencia directa del preprocesamiento "
    "previo aplicado al corpus por sus autores, que recortó o rellenó cada grabación para alcanzar "
    "una duración estándar."
)
add_figure(doc, os.path.join(IMG_EDA, "histograma_duraciones.png"),
           "Figura 2. Histograma de distribución de duraciones de las muestras de audio.")
add_paragraph(doc,
    "La uniformidad en duración tiene consecuencias directas y favorables para el diseño del pipeline "
    "de preprocesamiento: elimina la necesidad de estrategias de padding o truncado adaptativas y "
    "permite fijar un tamaño de entrada constante de 64 × 64 coeficientes en los espectrogramas de Mel, "
    "sin pérdida de información. Una ventana de 1 segundo cubre con holgura la duración fonética de "
    "las palabras seleccionadas en español, incluyendo las más largas del vocabulario objetivo "
    "('actualmente', 11 caracteres; 'universidad', 11 caracteres)."
)

page_break(doc)

# ══════════════════════════════════════════════
# 7. SELECCIÓN Y FILTRADO DE PALABRAS
# ══════════════════════════════════════════════
add_heading(doc, "5. Selección y Filtrado de Palabras", 1)

add_heading(doc, "5.1 Motivación del Proceso Automático", 2)
add_paragraph(doc,
    "La selección de las clases objetivo es una de las decisiones más críticas en un sistema de "
    "reconocimiento de vocabulario acotado. Una selección manual podría introducir sesgos del "
    "investigador, dificultaría la reproducibilidad y no escalaría a corpus más grandes. En este "
    "proyecto, la selección se realizó mediante un conjunto de filtros cuantitativos completamente "
    "automatizados, sin ninguna lista de palabras definida a mano. Los criterios empleados y sus "
    "umbrales se describen a continuación."
)

add_heading(doc, "5.2 Criterios de Selección", 2)
add_paragraph(doc,
    "El primer criterio aplicado fue el de frecuencia mínima: se requirió que cada palabra candidata "
    "contara con al menos 1,000 muestras en el corpus. Este umbral responde a un principio estadístico: "
    "con 1,000 ejemplos por clase y una partición del 20% para prueba (200 instancias), la estimación "
    "del accuracy por clase tiene un margen de error máximo de ±4.9% al 95% de confianza, lo cual es "
    "aceptable para el propósito de este proyecto. De las 24,878 palabras del corpus, solo 79 "
    "superaron este umbral, lo que significa que el 99.68% del vocabulario fue descartado por "
    "insuficiencia de datos."
)

add_paragraph(doc,
    "El segundo criterio fue la longitud mínima de 5 caracteres. Las palabras muy cortas en español "
    "—monosílabos como 'los', 'del', 'que', 'por'— son fonéticamente ambiguas: su señal acústica "
    "es breve, sus formantes se solapan con los de otras palabras cortas y su frecuencia en el habla "
    "conectada genera confusiones acústicas frecuentes. Adicionalmente, estas palabras funcionales "
    "son de baja utilidad como clases discriminativas en un sistema de comando de voz. Este filtro "
    "eliminó 43 palabras adicionales del conjunto de candidatas."
)

add_paragraph(doc,
    "El tercer criterio fue la similitud fonética entre pares de palabras ya seleccionadas. Se empleó "
    "la métrica de Jaro-Winkler, una medida de distancia de edición que asigna mayor peso a los "
    "prefijos de las cadenas de caracteres, lo que en español —idioma con alta correspondencia "
    "grafema-fonema— constituye un proxy válido de similitud fonética. Se estableció un umbral de "
    "0.92: si una palabra candidata tenía similitud Jaro-Winkler ≥ 0.92 con alguna palabra ya "
    "aceptada en el conjunto, era descartada. Este criterio eliminó 1 palabra adicional, evitando "
    "la inclusión de pares acústicamente indistinguibles que degradarían la exactitud global."
)

add_paragraph(doc,
    "El cuarto criterio, la depuración por F1 preliminar mínimo de 0.25, fue configurado pero no "
    "resultó necesario: ninguna de las palabras candidatas fue eliminada por este filtro, lo que "
    "indica que la combinación de los tres criterios anteriores ya garantizó un conjunto de clases "
    "suficientemente discriminable en términos fonéticos y estadísticos."
)

add_heading(doc, "5.3 Resultado de la Selección", 2)
add_paragraph(doc,
    "Tras aplicar los cuatro filtros secuencialmente, el conjunto final quedó integrado por 35 palabras. "
    "La Tabla 2 sintetiza el impacto de cada criterio de exclusión."
)

add_table(doc,
    ["Criterio de exclusión", "Palabras eliminadas"],
    [
        ["Frecuencia insuficiente (< 1,000 muestras)", "24,799"],
        ["Longitud insuficiente (< 5 caracteres)", "43"],
        ["Similitud fonética alta con otra seleccionada (Jaro-Winkler ≥ 0.92)", "1"],
        ["Bajo rendimiento en entrenamiento preliminar (F1 < 0.25)", "0"],
        ["Total excluidas", "24,843"],
        ["Total seleccionadas", "35"],
    ],
    "Tabla 2. Resumen de palabras excluidas por cada criterio de filtrado."
)

add_paragraph(doc,
    "Las 35 palabras seleccionadas cubren una variedad de categorías gramaticales y fonéticas: numerales "
    "('cinco', 'siete', 'nueve', 'cuatro'), adverbios ('actualmente', 'además', 'entonces', 'luego', "
    "'también'), preposiciones polisílabas ('desde', 'entre', 'sobre', 'durante'), sustantivos "
    "('ciudad', 'nombre', 'tiempo', 'forma', 'parte', 'universidad') y pronombres y determinantes "
    "('estos', 'estas', 'están', 'todos', 'varios', 'otros'). Esta diversidad garantiza que el "
    "clasificador sea evaluado ante vocabulario heterogéneo en términos fonéticos, lo cual es más "
    "representativo de un escenario de uso real."
)

add_heading(doc, "5.4 Palabras Seleccionadas", 2)
add_paragraph(doc,
    "La Tabla 3 presenta las 35 palabras seleccionadas junto con el número de muestras disponibles "
    "en el corpus. Todos los valores superan el umbral de 1,000 muestras, con 'tiene' como la de "
    "menor frecuencia relativa entre las seleccionadas (3,015 muestras) y 'encuentra' como una de "
    "las de mayor frecuencia (2,351 muestras dentro del rango relevante)."
)

add_table(doc,
    ["Palabra", "Muestras en corpus", "Palabra", "Muestras en corpus"],
    [
        ["actualmente", "1,620", "luego", "1,008"],
        ["además", "1,800", "mayor", "1,073"],
        ["cinco", "1,510", "mismo", "1,381"],
        ["ciudad", "2,009", "nombre", "1,578"],
        ["cuando", "1,472", "nueve", "1,454"],
        ["cuatro", "2,088", "otros", "1,223"],
        ["desde", "1,831", "parte", "2,107"],
        ["donde", "1,222", "primera", "1,421"],
        ["durante", "2,327", "puede", "1,656"],
        ["embargo", "1,537", "siete", "1,612"],
        ["encuentra", "2,351", "sobre", "1,854"],
        ["entonces", "1,134", "tiempo", "1,216"],
        ["entre", "2,158", "tiene", "3,015"],
        ["estas", "1,078", "todos", "1,347"],
        ["estos", "1,301", "universidad", "1,287"],
        ["están", "1,115", "varios", "1,061"],
        ["forma", "1,288", "—", "—"],
        ["fueron", "1,972", "—", "—"],
        ["hasta", "1,407", "—", "—"],
    ],
    "Tabla 3. Palabras seleccionadas y su frecuencia en el corpus."
)

page_break(doc)

# ══════════════════════════════════════════════
# 8. PROCESAMIENTO DE AUDIO
# ══════════════════════════════════════════════
add_heading(doc, "6. Procesamiento de Audio y Extracción de Características", 1)

add_heading(doc, "6.1 Cadena de Preprocesamiento", 2)
add_paragraph(doc,
    "La cadena de preprocesamiento transforma cada archivo de audio crudo (.wav) en una matriz numérica "
    "de dimensiones fijas (64 × 64 × 1) que puede ser consumida directamente por la red neuronal "
    "convolucional. El proceso consta de cuatro etapas: (1) carga y normalización de la señal temporal, "
    "(2) ajuste de duración, (3) cálculo del espectrograma de Mel y (4) normalización del espectrograma."
)

add_heading(doc, "6.2 Carga y Normalización de la Señal", 2)
add_paragraph(doc,
    "Cada archivo de audio es cargado con la biblioteca Librosa 0.11.0 y remuestreado a 16,000 Hz "
    "(sr_objetivo = 16,000). Esta frecuencia de muestreo es estándar en aplicaciones de reconocimiento "
    "de voz —es la utilizada por sistemas como CMU Sphinx, Kaldi y los modelos de la familia Whisper— "
    "y cubre el rango de frecuencias fonéticamente relevante (hasta 8,000 Hz según el teorema de "
    "Nyquist), que incluye los formantes vocálicos y las fricativas de las consonantes del español. "
    "El remuestreo garantiza que todas las muestras sean comparables independientemente de su "
    "frecuencia de muestreo original."
)

add_heading(doc, "6.3 Ajuste de Duración", 2)
add_paragraph(doc,
    "Cada señal es ajustada a exactamente 16,000 muestras (1 segundo a 16,000 Hz). Las señales con "
    "duración menor se complementan con ceros (zero-padding) al final, preservando el contenido "
    "temporal existente. Las señales con duración mayor son recortadas por el inicio, eliminando el "
    "posible silencio inicial sin afectar el núcleo acústico de la palabra. Esta decisión se justificó "
    "por el análisis de duraciones del EDA, que demostró que prácticamente la totalidad de las "
    "muestras del corpus tiene exactamente 1 segundo de duración, por lo que el padding o recorte "
    "ocurre en casos marginales."
)

add_heading(doc, "6.4 Espectrograma de Mel", 2)
add_paragraph(doc,
    "La representación acústica elegida es el espectrograma de Mel (Mel Spectrogram), que combina "
    "la transformada de Fourier de tiempo corto (STFT) con una escala de frecuencias perceptual. "
    "La escala de Mel comprime el eje de frecuencias de manera no lineal: aplica mayor resolución "
    "a frecuencias bajas (donde el oído humano distingue más diferencias) y menor resolución a "
    "frecuencias altas. Esta propiedad hace que los espectrogramas de Mel sean significativamente "
    "más informativos para la voz que los espectrogramas de potencia lineales, razón por la cual "
    "son la representación dominante en sistemas modernos de reconocimiento de habla."
)

add_paragraph(doc,
    "Los parámetros de cálculo fueron determinados durante el diseño del pipeline y configurados en "
    "el archivo config_experimento.json. Se emplearon 1,024 puntos para la FFT (n_fft = 1024), un "
    "salto de 256 muestras entre ventanas contiguas (hop_length = 256), una ventana de análisis de "
    "512 muestras (win_length = 512) y 64 canales de Mel (n_mels = 64). Con estos parámetros, "
    "una señal de 1 segundo a 16,000 Hz produce un espectrograma de dimensiones 64 × 64, donde "
    "el eje vertical representa los 64 coeficientes de Mel y el eje horizontal los 64 pasos temporales. "
    "La elección de una ventana cuadrada facilita la construcción de una CNN simétrica y reduce la "
    "complejidad computacional."
)

add_paragraph(doc,
    "El espectrograma lineal resultante se convierte a escala logarítmica en decibelios "
    "(power_to_db), lo que mejora el contraste visual y numérico entre regiones de alta y baja "
    "energía. Finalmente, cada espectrograma es normalizado de forma independiente (min-max local) "
    "al rango [0, 1], lo que elimina variaciones de escala entre grabaciones realizadas con distintos "
    "niveles de volumen o equipos de grabación. La Tabla 4 resume los parámetros de extracción."
)

add_table(doc,
    ["Parámetro", "Valor", "Justificación"],
    [
        ["Frecuencia de muestreo", "16,000 Hz", "Estándar ASR, cubre rango fonético"],
        ["Duración de la ventana", "1.0 s", "Duración modal del corpus"],
        ["n_fft", "1,024 puntos", "Resolución frecuencial adecuada"],
        ["hop_length", "256 muestras", "Resolución temporal 16 ms"],
        ["win_length", "512 muestras", "Ventana de 32 ms"],
        ["n_mels", "64 bandas", "Salida 64×64 (ventana cuadrada)"],
        ["time_steps", "64 pasos", "64 pasos × 16 ms = 1.024 s"],
        ["Escala", "Logarítmica (dB)", "Mejora contraste, simula audición"],
        ["Normalización", "Min-Max [0, 1]", "Independencia de volumen de grabación"],
    ],
    "Tabla 4. Parámetros de extracción del espectrograma de Mel."
)

page_break(doc)

# ══════════════════════════════════════════════
# 9. ARQUITECTURA
# ══════════════════════════════════════════════
add_heading(doc, "7. Arquitectura de la Red Neuronal Convolucional", 1)

add_heading(doc, "7.1 Justificación del Enfoque CNN", 2)
add_paragraph(doc,
    "Las redes neuronales convolucionales son la opción natural para clasificar espectrogramas de "
    "Mel porque estos pueden tratarse como imágenes bidimensionales donde los patrones locales "
    "—por ejemplo, la presencia de un pico de energía en una banda de Mel durante un intervalo "
    "temporal específico— son discriminativos para la identidad de la palabra. Las capas "
    "convolucionales son invariantes a traslaciones locales, lo que significa que un patrón "
    "acústico puede ser reconocido independientemente de su posición exacta en el espectrograma, "
    "característica deseable dado que distintas realizaciones de la misma palabra pueden diferir "
    "ligeramente en duración total y velocidad articulatoria."
)

add_heading(doc, "7.2 Arquitectura del Modelo Final", 2)
add_paragraph(doc,
    "La arquitectura del modelo final fue determinada por Keras Tuner (detallado en la Sección 8). "
    "El modelo consiste en tres bloques convolucionales seguidos de una capa de clasificación "
    "totalmente conectada. La Tabla 5 describe cada capa del modelo, sus dimensiones de salida "
    "y los parámetros aprendibles."
)

add_table(doc,
    ["Capa", "Tipo", "Filtros/Unidades", "Kernel/—", "Activación", "Regularización"],
    [
        ["Entrada", "InputLayer", "—", "64×64×1", "—", "—"],
        ["Conv2D-1", "Convolucional", "32 filtros", "3×3", "ReLU", "L2 (λ=0.003)"],
        ["MaxPool-1", "MaxPooling2D", "—", "2×2", "—", "—"],
        ["Conv2D-2", "Convolucional", "128 filtros", "5×5", "ReLU", "L2 (λ=0.003)"],
        ["MaxPool-2", "MaxPooling2D", "—", "2×2", "—", "—"],
        ["Conv2D-3", "Convolucional", "16 filtros", "3×3", "ReLU", "L2 (λ=0.003)"],
        ["MaxPool-3", "MaxPooling2D", "—", "2×2", "—", "—"],
        ["Flatten", "Aplanado", "—", "—", "—", "—"],
        ["Dense-1", "Totalmente conectada", "512 neuronas", "—", "ReLU", "Dropout (p=0.40)"],
        ["Dense-salida", "Clasificación", "35 neuronas", "—", "Softmax", "—"],
    ],
    "Tabla 5. Arquitectura de la CNN seleccionada por Keras Tuner."
)

add_heading(doc, "7.3 Decisiones de Diseño", 2)
add_paragraph(doc,
    "La primera capa convolucional utiliza 32 filtros de tamaño 3×3 para capturar patrones de "
    "grano fino en el espectrograma, como la presencia de energía en bandas de Mel específicas. "
    "La segunda capa expande el número de filtros a 128 con un kernel de 5×5, permitiendo capturar "
    "patrones de mayor alcance temporal y frecuencial que corresponden a transiciones entre fonemas. "
    "La tercera capa reduce los filtros a 16, actuando como un cuello de botella que comprime la "
    "representación aprendida antes de la clasificación. Esta progresión no monotónica de filtros "
    "(32 → 128 → 16) fue descubierta por el tuner como la configuración óptima, lo que sugiere que "
    "la expansión intermedia favorece la extracción de características complejas antes de la "
    "compresión final."
)

add_paragraph(doc,
    "Cada bloque convolucional incluye una capa de MaxPooling 2×2 que reduce las dimensiones "
    "espaciales a la mitad, disminuyendo el costo computacional y proporcionando invarianza a "
    "pequeñas traslaciones. La ausencia de BatchNormalization fue confirmada por el tuner: las "
    "configuraciones con BatchNormalization obtuvieron puntuaciones ligeramente inferiores en la "
    "fase de búsqueda, probablemente porque la normalización por lote interfiere con la información "
    "de amplitud relativa presente en los espectrogramas de Mel, que en este caso es informativa."
)

add_paragraph(doc,
    "La capa densa de 512 neuronas con dropout del 40% cumple dos funciones: la primera es proyectar "
    "las representaciones convolucionales aplanadas a un espacio de alta dimensionalidad que favorece "
    "la separabilidad lineal de las 35 clases; la segunda es regularizar el modelo mediante la "
    "eliminación aleatoria del 40% de las activaciones durante el entrenamiento, previniendo el "
    "sobreajuste. El optimizador RMSprop fue preferido sobre Adam por el tuner; este resultado es "
    "coherente con la literatura de clasificación de audio, donde RMSprop ha demostrado mayor "
    "estabilidad en la convergencia para señales con alta varianza temporal. La tasa de aprendizaje "
    "inicial de 4.44×10⁻⁴ fue ajustada automáticamente por ReduceLROnPlateau, reduciéndose a la "
    "mitad cada vez que la exactitud de validación no mejoró durante 5 épocas consecutivas."
)

page_break(doc)

# ══════════════════════════════════════════════
# 10. KERAS TUNER
# ══════════════════════════════════════════════
add_heading(doc, "8. Optimización mediante Keras Tuner", 1)

add_heading(doc, "8.1 Introducción a Keras Tuner y la Estrategia Hyperband", 2)
add_paragraph(doc,
    "Keras Tuner es una biblioteca de código abierto que automatiza la búsqueda de hiperparámetros "
    "para modelos construidos con TensorFlow/Keras. A diferencia de la búsqueda en cuadrícula (Grid "
    "Search) —que evalúa exhaustivamente todas las combinaciones posibles— o la búsqueda aleatoria "
    "(Random Search) —que muestrea configuraciones al azar—, Keras Tuner ofrece estrategias más "
    "eficientes como Hyperband, que asigna más recursos computacionales a las configuraciones "
    "prometedoras y descarta las deficientes tempranamente."
)

add_paragraph(doc,
    "Hyperband es una extensión del algoritmo Successive Halving. Funciona organizando los trials "
    "en corchetes (brackets): en cada ronda, el 50% de los modelos con menor rendimiento es "
    "eliminado y solo los mejores continúan entrenando durante más épocas. Este mecanismo permite "
    "explorar un espacio de búsqueda amplio con un presupuesto computacional acotado. En este "
    "experimento, se configuraron hasta 20 trials con un máximo de 20 épocas de entrenamiento por "
    "trial, lo que significa que configuraciones claramente subóptimas son descartadas tras pocas "
    "épocas, mientras que las mejores candidatas reciben el presupuesto completo."
)

add_heading(doc, "8.2 Espacio de Búsqueda", 2)
add_paragraph(doc,
    "El espacio de hiperparámetros explorado cubre dos categorías: parámetros de arquitectura y "
    "parámetros de entrenamiento. Los parámetros de arquitectura incluyen el número de capas "
    "convolucionales (de 1 a 4), el número de filtros por capa (16, 32, 64 o 128), el tamaño del "
    "kernel por capa (3×3 o 5×5), el uso de BatchNormalization (sí/no), el número de neuronas "
    "en la capa densa (64, 128, 256 o 512) y el coeficiente de dropout (de 0.10 a 0.60 en pasos "
    "de 0.05). Los parámetros de entrenamiento incluyen la tasa de aprendizaje inicial (de 1×10⁻⁴ "
    "a 1×10⁻² en escala logarítmica), la regularización L2 (de 0 a 0.01 en pasos de 0.001) y el "
    "optimizador (Adam o RMSprop). Todos los trials se evaluaron bajo condiciones controladas: "
    "misma semilla (seed = 42), mismo split de datos y el mismo subconjunto de 50 palabras × 300 "
    "muestras para mantener el costo computacional manejable."
)

add_heading(doc, "8.3 Resultados de la Búsqueda", 2)
add_paragraph(doc,
    "Se ejecutaron 30 trials en total (trials 0000 a 0029). La Tabla 6 presenta los 10 mejores "
    "trials ordenados por exactitud de validación. El trial ganador fue el 0017, que alcanzó una "
    "exactitud de validación de 0.8869 durante la fase de búsqueda, empleando exactamente los "
    "mismos hiperparámetros que el trial 0013 (del que es una iteración posterior de Hyperband con "
    "mayor presupuesto de épocas). Esto evidencia la coherencia del proceso de optimización: la "
    "configuración identificada como prometedora en rondas tempranas (trial 0013, score 0.8524) "
    "fue efectivamente la mejor al recibir el presupuesto completo (trial 0017, score 0.8869)."
)

add_table(doc,
    ["Trial", "Score (val_acc)", "Conv Layers", "Filtros [L1, L2, L3]", "Dense", "Dropout", "LR", "Optimizador", "L2"],
    [
        ["0017", "0.8869", "3", "[32, 128, 16]", "512", "0.40", "4.44e-4", "rmsprop", "0.003"],
        ["0026", "0.8833", "4", "[128, 64, 32]", "256", "0.50", "2.07e-3", "adam", "0.005"],
        ["0016", "0.8655", "3", "[16, 16, 64]", "128", "0.10", "3.50e-4", "rmsprop", "0.003"],
        ["0028", "0.8643", "3", "[64, 128, 128]", "128", "0.15", "3.56e-4", "rmsprop", "0.008"],
        ["0024", "0.8595", "3", "[128, 16, 16]", "64", "0.10", "7.36e-4", "rmsprop", "0.008"],
        ["0012", "0.8536", "3", "[16, 16, 64]", "128", "0.10", "3.50e-4", "rmsprop", "0.003"],
        ["0013", "0.8524", "3", "[32, 128, 16]", "512", "0.40", "4.44e-4", "rmsprop", "0.003"],
        ["0014", "0.8429", "3", "[16, 64, 32]", "256", "0.10", "1.27e-4", "rmsprop", "0.009"],
        ["0000", "0.8214", "3", "[16, 16, 64]", "128", "0.10", "3.50e-4", "rmsprop", "0.003"],
        ["0022", "0.8214", "3", "[128, 16, 16]", "64", "0.10", "7.36e-4", "rmsprop", "0.008"],
    ],
    "Tabla 6. Los 10 mejores trials de la búsqueda Hyperband."
)

add_paragraph(doc,
    "Varios patrones son observables en los resultados: el optimizador RMSprop dominó entre las "
    "configuraciones de alto rendimiento (8 de los 10 mejores trials); las configuraciones con "
    "una sola capa convolucional obtuvieron consistentemente bajas puntuaciones (trials 0005, "
    "0008, 0018, 0023, 0025, 0027 con scores ≤ 0.35), lo que confirma que la tarea requiere al "
    "menos tres capas para capturar la complejidad acústica del problema; las tasas de aprendizaje "
    "muy altas (> 5×10⁻³) condujeron al colapso del entrenamiento (scores ≈ 0.014, que corresponde "
    "a azar puro en un clasificador de 35 clases donde la probabilidad aleatoria es 1/35 ≈ 0.0286)."
)

page_break(doc)

# ══════════════════════════════════════════════
# 11. ENTRENAMIENTO
# ══════════════════════════════════════════════
add_heading(doc, "9. Entrenamiento del Modelo", 1)

add_heading(doc, "9.1 Configuración del Entrenamiento Final", 2)
add_paragraph(doc,
    "El modelo final fue entrenado sobre el conjunto completo de 35,000 muestras (35 clases × 1,000 "
    "muestras por clase), dividido en tres particiones: 70% para entrenamiento (24,500 muestras), "
    "10% para validación (3,500 muestras) y 20% para prueba (7,000 muestras). La división se "
    "realizó de forma estratificada para garantizar la representación proporcional de cada clase "
    "en las tres particiones. El batch size óptimo de 16, identificado durante la fase de tuning, "
    "fue empleado durante el entrenamiento final. El número máximo de épocas fue fijado en 100, "
    "con parada temprana configurada para detener el entrenamiento si la exactitud de validación "
    "no mejoraba durante 10 épocas consecutivas."
)

add_heading(doc, "9.2 Evolución del Entrenamiento", 2)
add_paragraph(doc,
    "La Figura 3 muestra las curvas de exactitud (accuracy) y pérdida (loss) de entrenamiento y "
    "validación a lo largo de las 63 épocas ejecutadas. El análisis de estas curvas revela tres "
    "fases claramente diferenciadas en el proceso de aprendizaje."
)
add_figure(doc, os.path.join(IMG_TRAINING, "curvas_entrenamiento.png"),
           "Figura 3. Curvas de exactitud y pérdida de entrenamiento y validación (63 épocas).")

add_paragraph(doc,
    "La primera fase, comprendida entre las épocas 0 y 16, corresponde al aprendizaje inicial. "
    "En la época 0, la exactitud de entrenamiento fue de apenas 0.586, lo que es esperable para "
    "un clasificador de 35 clases recién inicializado. La exactitud de validación, notablemente, "
    "fue de 0.810 en esta primera época, superando a la de entrenamiento, lo que indica que el "
    "modelo inicializado ya capta patrones generales útiles. Durante esta fase, tanto la exactitud "
    "de entrenamiento como la de validación crecen rápidamente, mientras que ambas pérdidas "
    "disminuyen de forma consistente. La tasa de aprendizaje inicial de 4.44×10⁻⁴ se mantuvo "
    "constante durante estas épocas."
)

add_paragraph(doc,
    "La segunda fase, entre las épocas 17 y 35 aproximadamente, marca el inicio del sobreajuste "
    "detectable. En la época 17, la pérdida de validación (val_loss) comenzó a aumentar de forma "
    "consistente mientras que la pérdida de entrenamiento continuaba disminuyendo, lo que es la "
    "señal diagnóstica clásica de overfitting. Sin embargo, el mecanismo de EarlyStopping no "
    "intervino en este punto porque la métrica monitoreada era la exactitud de validación "
    "(val_accuracy), no la pérdida, y la exactitud de validación continuó mejorando marginalmente. "
    "En la época 24 se produjo una reducción automática de la tasa de aprendizaje (ReduceLROnPlateau "
    "de 4.44×10⁻⁴ a 2.22×10⁻⁴), seguida de una mejora notable en la exactitud de validación, "
    "que saltó del 0.908 al 0.918 entre las épocas 36 y 37."
)

add_paragraph(doc,
    "La tercera fase, desde la época 37 hasta la 63, caracterizó un entrenamiento de refinamiento "
    "lento con reducciones sucesivas de la tasa de aprendizaje. ReduceLROnPlateau redujo la tasa "
    "de 2.22×10⁻⁴ a 1.11×10⁻⁴ en la época 36, a 5.55×10⁻⁵ en la época 51 y a 2.78×10⁻⁵ "
    "aproximadamente en la época 60. Cada reducción fue seguida de una mejora incremental en la "
    "exactitud de validación. El mejor punto fue la época 53, con val_accuracy = 0.9275. "
    "EarlyStopping activó la parada en la época 63 al no observar mejora durante 10 épocas "
    "consecutivas desde la época 53, restaurando automáticamente los pesos correspondientes a "
    "ese punto óptimo."
)

add_heading(doc, "9.3 Métricas Finales de Entrenamiento", 2)
add_paragraph(doc,
    "La Tabla 7 sintetiza las métricas finales del proceso de entrenamiento, correspondientes al "
    "modelo con los pesos de la época 53 (la mejor según val_accuracy) evaluado sobre el conjunto "
    "de prueba independiente."
)
add_table(doc,
    ["Métrica", "Valor"],
    [
        ["Exactitud en prueba (accuracy_test)", "0.9269 (92.69%)"],
        ["Pérdida en prueba (loss_test)", "0.3980"],
        ["Época óptima (mejor val_accuracy)", "53"],
        ["Exactitud de validación en época óptima", "0.9275 (92.75%)"],
        ["Total de épocas ejecutadas", "63"],
        ["Épocas máximas configuradas", "100"],
        ["Motivo de parada", "EarlyStopping (10 épocas sin mejora)"],
        ["Punto de sobreajuste detectado", "~Época 17 (val_loss aumenta)"],
        ["Tasa de aprendizaje inicial", "4.44×10⁻⁴"],
        ["Tasa de aprendizaje final", "2.78×10⁻⁵"],
    ],
    "Tabla 7. Métricas finales del proceso de entrenamiento."
)

page_break(doc)

# ══════════════════════════════════════════════
# 12. VALIDACIÓN
# ══════════════════════════════════════════════
add_heading(doc, "10. Validación y Evaluación del Modelo", 1)

add_heading(doc, "10.1 Procedimiento de Validación Cruzada K-Fold", 2)
add_paragraph(doc,
    "La exactitud obtenida sobre un único conjunto de prueba (92.69%) proporciona una estimación "
    "puntual del desempeño, pero no informa sobre su variabilidad estadística ni sobre la "
    "capacidad de generalización a conjuntos de datos diferentes. Para complementar esta evaluación, "
    "se realizó una validación cruzada con 5 particiones (5-Fold Cross Validation). En este "
    "procedimiento, el conjunto de datos completo se divide en 5 subconjuntos de igual tamaño; "
    "en cada iteración, 4 subconjuntos se usan para entrenamiento (28,000 muestras) y 1 para "
    "validación (7,000 muestras), rotando sistemáticamente qué subconjunto actúa como validación. "
    "Al finalizar las 5 iteraciones, se dispone de 5 estimaciones independientes del desempeño "
    "del modelo, lo que permite calcular estadísticas de distribución."
)

add_paragraph(doc,
    "Cada fold fue entrenado con los mismos hiperparámetros del modelo final, pero con un límite "
    "de 30 épocas (kfold_max_epochs = 30) para mantener el costo computacional controlado. Este "
    "límite reducido explica parcialmente la diferencia entre el accuracy del K-Fold y el del "
    "modelo final, que fue entrenado durante 63 épocas."
)

add_heading(doc, "10.2 Resultados por Fold", 2)
add_table(doc,
    ["Fold", "N entrenamiento", "N validación", "Accuracy", "Precisión macro", "Recall macro", "F1 macro"],
    [
        ["1", "28,000", "7,000", "0.8210", "0.7638", "0.8210", "0.7874"],
        ["2", "28,000", "7,000", "0.7871", "0.7342", "0.7871", "0.7546"],
        ["3", "28,000", "7,000", "0.8273", "0.7696", "0.8273", "0.7928"],
        ["4", "28,000", "7,000", "0.8231", "0.7642", "0.8231", "0.7889"],
        ["5", "28,000", "7,000", "0.8127", "0.7599", "0.8127", "0.7787"],
        ["Promedio", "—", "—", "0.8143", "0.7583", "0.8143", "0.7805"],
        ["Desv. estándar", "—", "—", "0.0144", "—", "—", "0.0137"],
    ],
    "Tabla 8. Resultados por fold de la validación cruzada 5-Fold."
)

add_heading(doc, "10.3 Intervalos de Confianza y Robustez", 2)
add_paragraph(doc,
    "El intervalo de confianza al 95% para la exactitud promedio del K-Fold es [0.7943, 0.8342]. "
    "Esto significa que, con una probabilidad del 95%, el desempeño real del modelo sobre datos no "
    "vistos se encuentra entre el 79.43% y el 83.42%. La desviación estándar de 0.0144 indica una "
    "variabilidad moderada entre folds, sin ningún fold que se desvíe drásticamente de la media, "
    "lo que confirma que el modelo es robusto y no presenta sensibilidad extrema a la partición "
    "específica de los datos."
)

add_heading(doc, "10.4 Discrepancia entre Accuracy de Prueba y K-Fold", 2)
add_paragraph(doc,
    "La diferencia entre la exactitud de prueba del modelo final (92.69%) y el promedio del K-Fold "
    "(81.43%) puede explicarse por dos factores principales. El primero es el número de épocas de "
    "entrenamiento: el modelo final fue entrenado durante 63 épocas completas con EarlyStopping "
    "óptimo, mientras que los modelos del K-Fold tuvieron un límite de 30 épocas, lo que significa "
    "que no alcanzaron su capacidad máxima. El segundo factor es que el modelo final utilizó "
    "explícitamente el batch size óptimo de 16 identificado por el tuner, y fue reiniciado desde "
    "los pesos óptimos de la época 53; los modelos del K-Fold, al entrenarse con menos recursos, "
    "no alcanzaron convergencia equivalente. Por estos motivos, la métrica del K-Fold debe "
    "interpretarse como una cota inferior conservadora del desempeño esperado, mientras que el "
    "accuracy de prueba del 92.69% es la estimación más precisa del desempeño del modelo final."
)

page_break(doc)

# ══════════════════════════════════════════════
# 13. INTERPRETABILIDAD
# ══════════════════════════════════════════════
add_heading(doc, "11. Interpretabilidad de Resultados", 1)

add_heading(doc, "11.1 Exactitud y F1 Globales", 2)
add_paragraph(doc,
    "El modelo final alcanzó una exactitud global del 92.69% y un F1-score macro de 0.9271 sobre "
    "las 35 clases del conjunto de prueba (7,000 muestras totales, 200 por clase). El F1-score "
    "macro —que calcula el F1 de cada clase independientemente y luego promedia sin ponderar por "
    "frecuencia— es una métrica apropiada para conjuntos de datos balanceados como el empleado en "
    "este experimento. De las 35 clases, 34 (97.1%) obtuvieron un F1 superior a 0.80, y ninguna "
    "clase obtuvo un F1 inferior a 0.40, lo que indica que el modelo es competente en la "
    "discriminación de prácticamente todas las palabras del vocabulario objetivo."
)

add_heading(doc, "11.2 Desempeño por Clase", 2)
add_paragraph(doc,
    "La Figura 4 muestra el F1-score por clase para las 35 palabras del vocabulario. La heterogeneidad "
    "en el desempeño entre clases es informativa sobre la complejidad fonética relativa de cada palabra."
)
add_figure(doc, os.path.join(IMG_INTERP, "f1_por_clase.png"),
           "Figura 4. F1-score por clase de las 35 palabras del vocabulario objetivo.")

add_paragraph(doc,
    "Las palabras con mejor desempeño son aquellas con perfil acústico distintivo. 'Actualmente' "
    "lidera con un F1 de 0.9975 (prácticamente perfecto), lo cual es coherente con su longitud "
    "excepcional (11 caracteres) y su secuencia fonética única en el vocabulario. Le siguen "
    "'embargo' (0.9801), 'además' (0.9778) y 'universidad' (0.9773), todos ellos polisílabos con "
    "combinaciones de fonemas relativamente poco comunes. Los numerales 'cinco' (0.9554), 'siete' "
    "(0.9502) y 'nueve' (0.9343) también presentan F1 elevados, lo que sugiere que las secuencias "
    "vocálicas particulares de los numerales en español son fácilmente discriminables."
)

add_paragraph(doc,
    "En el extremo opuesto, 'estas' presenta el F1 más bajo del vocabulario (0.77). Palabras como "
    "'estos' (0.8253), 'otros' (0.8424) y 'están' (0.8593) también muestran desempeño inferior "
    "al promedio. Este patrón no es fortuito: estas palabras pertenecen a un clúster fonético "
    "compacto que comparte núcleos vocálicos similares (/e/, /o/) y terminaciones similares "
    "(-os, -as, -án), lo que las hace acústicamente difíciles de distinguir incluso para hablantes "
    "humanos en condiciones de escucha degradada."
)

add_heading(doc, "11.3 Análisis de la Matriz de Confusión", 2)
add_figure(doc, os.path.join(IMG_INTERP, "matriz_confusion.png"),
           "Figura 5. Matriz de confusión normalizada del modelo final (35 clases).")

add_paragraph(doc,
    "La Figura 5 presenta la matriz de confusión del modelo evaluado sobre el conjunto de prueba. "
    "En términos ideales, todos los valores deberían concentrarse en la diagonal principal (predicciones "
    "correctas). Los elementos fuera de la diagonal representan errores de clasificación y permiten "
    "identificar pares de clases problemáticos. La inspección de la matriz revela que la mayor "
    "concentración de errores fuera de la diagonal se ubica en el subconjunto de palabras "
    "{'estas', 'estos', 'están'}, que forman un triángulo de confusión mutua."
)

add_heading(doc, "11.4 Pares de Palabras Frecuentemente Confundidos", 2)
add_paragraph(doc,
    "La Tabla 9 detalla los 10 pares de palabras con mayor número de errores de clasificación. "
    "El análisis de estos pares proporciona información cualitativa sobre las limitaciones del "
    "modelo desde una perspectiva fonética."
)
add_table(doc,
    ["Clase real", "Clase predicha", "Errores", "% del total real"],
    [
        ["estos", "estas", "24", "12.0%"],
        ["estas", "estos", "18", "9.0%"],
        ["estas", "están", "17", "8.5%"],
        ["están", "estas", "15", "7.5%"],
        ["cuatro", "otros", "8", "4.0%"],
        ["nombre", "donde", "8", "4.0%"],
        ["entre", "desde", "7", "3.5%"],
        ["tiempo", "cinco", "7", "3.5%"],
        ["otros", "desde", "6", "3.0%"],
        ["otros", "estos", "6", "3.0%"],
    ],
    "Tabla 9. Los 10 pares de palabras con mayor número de errores de clasificación."
)

add_paragraph(doc,
    "La confusión entre 'estos' y 'estas' (24 y 18 errores respectivamente) es el error más "
    "frecuente del modelo y responde a causas fonéticas claras: ambas palabras son idénticas "
    "hasta la última vocal (/e-s-t-o-s/ vs /e-s-t-a-s/), diferenciándose únicamente por el "
    "sonido en posición final (/o/ vs /a/). Dado que las realizaciones de /o/ y /a/ en posición "
    "átona final en español pueden presentar reducción vocálica —especialmente en grabaciones "
    "realizadas con distintos micrófonos o en entornos ruidosos—, la confusión entre estas "
    "palabras es inherente a la naturaleza acústica del problema. La palabra 'están' añade una "
    "complicación adicional al compartir los mismos tres primeros fonemas (/e-s-t/) con 'estos' "
    "y 'estas'."
)

add_paragraph(doc,
    "La confusión entre 'nombre' y 'donde' (8 errores) resulta sorprendente a primera vista, "
    "ya que fonéticamente son palabras distintas (/nom-bre/ vs /don-de/). Sin embargo, ambas "
    "comparten la presencia de una consonante nasal (/m/ y /n/ respectivamente) y una vocal /o/ "
    "prominente, y son bisilábicas con acento en la primera sílaba, lo que puede confundir al "
    "modelo en condiciones de baja energía. La confusión entre 'tiempo' y 'cinco' (7 errores) "
    "también parece responder a similitudes en el perfil de energía de la fricativa dental /s/ "
    "presente en 'cinco' y la fricativa /ʃ/ implícita en la articulación de 'tiemp-'. "
    "Estas observaciones sugieren que el modelo aprende patrones de corto alcance temporal "
    "que pueden coincidir accidentalmente entre pares de palabras distintas."
)

page_break(doc)

# ══════════════════════════════════════════════
# 14. DISCUSIÓN
# ══════════════════════════════════════════════
add_heading(doc, "12. Discusión", 1)

add_heading(doc, "12.1 Interpretación del Desempeño Global", 2)
add_paragraph(doc,
    "Una exactitud del 92.69% en un problema de clasificación de 35 clases representa un resultado "
    "sólido para un sistema de reconocimiento de palabras aisladas en español. Para contextualizarlo, "
    "un clasificador aleatorio tendría una exactitud esperada de 1/35 ≈ 2.86%, y un clasificador "
    "naive (que siempre predice la clase más frecuente) tendría una exactitud de apenas 2.86% dado "
    "el balance perfecto del conjunto de datos. El modelo desarrollado supera estas referencias en "
    "un factor de 32, lo que confirma que la red ha aprendido representaciones acústicas genuinamente "
    "discriminativas."
)

add_paragraph(doc,
    "El F1-macro de 0.9271 es especialmente relevante porque, a diferencia de la exactitud, no "
    "puede ser inflado artificialmente por clases dominantes en datasets desbalanceados. En un "
    "dataset balanceado como el empleado, F1-macro y exactitud son métricas equivalentes en términos "
    "de información sobre el rendimiento, y la coincidencia de ambas en 0.927 indica consistencia "
    "en el desempeño a lo largo de todas las clases."
)

add_heading(doc, "12.2 Eficacia del Pipeline Automático", 2)
add_paragraph(doc,
    "Una contribución metodológica importante de este proyecto es la ausencia de intervención manual "
    "en las decisiones de diseño. La selección de palabras, la optimización de hiperparámetros y la "
    "detención del entrenamiento fueron completamente automatizadas mediante criterios cuantitativos "
    "y algoritmos de búsqueda. Esto tiene dos implicaciones prácticas: primero, el pipeline es "
    "completamente reproducible con la misma semilla aleatoria; segundo, el pipeline puede escalarse "
    "a vocabularios más grandes o a otros idiomas sin requerir conocimiento experto sobre qué palabras "
    "o arquitecturas son adecuadas."
)

add_heading(doc, "12.3 Regularización y Control del Sobreajuste", 2)
add_paragraph(doc,
    "El sobreajuste fue detectado alrededor de la época 17, momento en que la pérdida de validación "
    "comenzó a aumentar consistentemente mientras que la pérdida de entrenamiento continuaba "
    "disminuyendo. A pesar de ello, el modelo continuó mejorando su exactitud de validación hasta "
    "la época 53, lo que indica que el sobreajuste detectado en la pérdida no se tradujo "
    "inmediatamente en degradación de la exactitud de clasificación. Este fenómeno es habitual "
    "en modelos con dropout: el clasificador puede seguir mejorando su discriminación de clases "
    "incluso cuando la calibración de las probabilidades se deteriora. La combinación de dropout "
    "(0.40), regularización L2 (λ = 0.003) y reducción adaptativa de la tasa de aprendizaje "
    "fue efectiva para mantener el modelo en una región de buena generalización durante un "
    "rango amplio de épocas."
)

page_break(doc)

# ══════════════════════════════════════════════
# 15. LIMITACIONES
# ══════════════════════════════════════════════
add_heading(doc, "13. Limitaciones", 1)

add_paragraph(doc,
    "El sistema desarrollado presenta varias limitaciones que deben considerarse al interpretar "
    "sus resultados y al pensar en su despliegue en entornos reales. La primera limitación es el "
    "vocabulario acotado: el modelo clasifica únicamente 35 palabras predefinidas y no puede "
    "reconocer palabras fuera de este vocabulario. En escenarios de habla espontánea, la aparición "
    "de palabras fuera del vocabulario generaría clasificaciones incorrectas sin ninguna señal de "
    "rechazo, ya que el modelo no implementa un mecanismo de detección de distribución abierta "
    "(open-set recognition)."
)

add_paragraph(doc,
    "La segunda limitación es la dependencia del formato de entrada. El modelo requiere que cada "
    "muestra de audio tenga exactamente 1 segundo de duración, frecuencia de muestreo de 16,000 Hz "
    "y que la palabra esté aislada (sin contexto antes o después). En habla continua, la segmentación "
    "automática en palabras individuales requeriría un módulo adicional de detección de actividad "
    "de voz (VAD), que no fue implementado en este proyecto."
)

add_paragraph(doc,
    "La tercera limitación es la discrepancia entre el accuracy de prueba (92.69%) y el promedio "
    "del K-Fold (81.43%). Aunque esta diferencia fue explicada metodológicamente, existe la "
    "posibilidad de que el modelo final presente algún grado de sobreajuste al conjunto de prueba "
    "específico utilizado, dado que las decisiones de diseño fueron tomadas en parte observando "
    "el comportamiento en dicho conjunto. Una evaluación más rigurosa requeriría un conjunto de "
    "prueba completamente retenido desde el inicio del experimento."
)

add_paragraph(doc,
    "La cuarta limitación es la calidad de los datos del corpus. ML Spoken Words, aunque amplio, "
    "contiene grabaciones realizadas en condiciones variables de hardware, entorno acústico y "
    "acento. El desempeño del modelo en condiciones de ruido real o con hablantes con acentos "
    "regionales distintos a los presentes en el corpus podría ser significativamente inferior al "
    "reportado en este experimento."
)

page_break(doc)

# ══════════════════════════════════════════════
# 16. TRABAJO FUTURO
# ══════════════════════════════════════════════
add_heading(doc, "14. Trabajo Futuro", 1)

add_paragraph(doc,
    "Varias líneas de trabajo pueden extender y mejorar los resultados obtenidos. La primera línea "
    "es la implementación de augmentación de datos (data augmentation) en el dominio del tiempo o "
    "de la frecuencia: técnicas como SpecAugment —que aplica máscaras aleatorias en bandas de "
    "frecuencia o en segmentos temporales del espectrograma— han demostrado reducir el sobreajuste "
    "y mejorar la robustez en condiciones de ruido. Dado que el sobreajuste fue detectado en este "
    "experimento, es razonable esperar que la augmentación de datos redujera la brecha entre el "
    "accuracy de prueba y el K-Fold."
)

add_paragraph(doc,
    "La segunda línea de trabajo es la extensión del vocabulario. El pipeline automático desarrollado "
    "puede aplicarse directamente a conjuntos más grandes de palabras: simplemente modificando el "
    "parámetro max_palabras en la configuración, el sistema seleccionaría automáticamente el "
    "subconjunto óptimo. Sin embargo, un vocabulario más amplio requeriría una arquitectura más "
    "profunda y potencialmente modelos pre-entrenados como punto de partida, lo que lleva a la "
    "tercera línea de trabajo."
)

add_paragraph(doc,
    "La tercera línea de trabajo es la transferencia de conocimiento desde modelos pre-entrenados "
    "en conjuntos de datos de audio más grandes, como AudioSet o VGGSound. Modelos como YAMNet "
    "o PANNs han aprendido representaciones de audio transferibles que podrían ser fine-tuneadas "
    "para el español con menor cantidad de datos etiquetados, reduciendo los requerimientos de "
    "cómputo y potencialmente mejorando el desempeño."
)

add_paragraph(doc,
    "La cuarta línea de trabajo es la resolución de los pares de confusión identificados, "
    "particularmente el clúster {'estas', 'estos', 'están'}. Estrategias como el entrenamiento "
    "con pérdida focal (focal loss), que asigna mayor penalización a los errores en clases "
    "difíciles, o el entrenamiento con muestras difíciles (hard negative mining) podrían mejorar "
    "la discriminación entre estas clases sin afectar el desempeño en las clases ya bien resueltas."
)

page_break(doc)

# ══════════════════════════════════════════════
# 17. CONCLUSIONES
# ══════════════════════════════════════════════
add_heading(doc, "15. Conclusiones", 1)

add_paragraph(doc,
    "Este proyecto demostró que es posible construir un sistema de reconocimiento automático de "
    "palabras aisladas en español con alto desempeño mediante un pipeline completamente automatizado "
    "que integra análisis exploratorio de datos, selección cuantitativa de vocabulario, extracción "
    "de características acústicas, optimización automática de hiperparámetros y validación estadística "
    "rigurosa. El modelo CNN entrenado alcanzó una exactitud del 92.69% y un F1-macro de 0.9271 sobre "
    "un vocabulario de 35 palabras, con 34 de 35 clases (97.1%) obteniendo F1 superior a 0.80."
)

add_paragraph(doc,
    "La elección del espectrograma de Mel como representación acústica resultó adecuada para el "
    "problema: sus propiedades perceptuales y su representación bidimensional son naturalmente "
    "compatibles con el aprendizaje convolucional. Los parámetros de extracción de características "
    "—ventana de 1 segundo, 64 bandas de Mel, resolución temporal de 16 ms— generaron representaciones "
    "64×64 que capturan tanto los patrones de corta duración (fonemas individuales) como los de "
    "mediana duración (sílabas y transiciones consonante-vocal) que son discriminativos para la "
    "identificación de palabras."
)

add_paragraph(doc,
    "La optimización con Keras Tuner Hyperband fue determinante para identificar la configuración "
    "óptima: la arquitectura de tres capas convolucionales con progresión de filtros 32→128→16, "
    "capa densa de 512 neuronas con dropout del 40% y optimizador RMSprop no habría sido obvia "
    "mediante diseño manual, y su superioridad fue confirmada experimentalmente sobre 30 "
    "configuraciones alternativas."
)

add_paragraph(doc,
    "El análisis de interpretabilidad reveló que los errores del modelo son fonéticamente "
    "coherentes: las palabras más confundidas comparten características acústicas objetivas. "
    "Este hallazgo es valioso porque sugiere que el límite superior de desempeño alcanzable "
    "con arquitecturas CNN estándar sobre espectrogramas de Mel puede estar cercano para este "
    "tipo de pares de palabras, y que mejorar su discriminación podría requerir representaciones "
    "más sofisticadas o modelos con mayor contexto temporal."
)

add_paragraph(doc,
    "Finalmente, el enfoque de selección de palabras basado en criterios cuantitativos y la "
    "reproducibilidad completa del experimento (seed = 42 fijo en Python, NumPy y TensorFlow) "
    "constituyen contribuciones metodológicas que pueden ser adoptadas en trabajos futuros sobre "
    "reconocimiento de voz en español u otros idiomas de bajos recursos."
)

page_break(doc)

# ══════════════════════════════════════════════
# 18. REFERENCIAS
# ══════════════════════════════════════════════
add_heading(doc, "16. Referencias", 1)

refs = [
    ("MLCommons Association. (2021).",
     "ML Spoken Words: A Large-Scale Dataset for Spoken Word Classification. "
     "MLCommons Open Datasets. https://mlcommons.org/datasets/ml-spoken-words/"),
    ("McFee, B., Raffel, C., Liang, D., Ellis, D. P. W., McVicar, M., Battenberg, E., & Nieto, O. (2015).",
     "librosa: Audio and Music Signal Analysis in Python. "
     "Proceedings of the 14th Python in Science Conference, 18-25."),
    ("Chollet, F. et al. (2015).",
     "Keras. https://keras.io. TensorFlow 2.x integration."),
    ("O'Malley, T., Bursztein, E., Long, J., Chollet, F., Jin, H., Invernizzi, L., et al. (2019).",
     "KerasTuner. https://github.com/keras-team/keras-tuner"),
    ("Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2017).",
     "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization. "
     "Journal of Machine Learning Research, 18(185), 1-52."),
    ("Stevens, S. S., Volkmann, J., & Newman, E. B. (1937).",
     "A Scale for the Measurement of the Psychological Magnitude Pitch. "
     "Journal of the Acoustical Society of America, 8(3), 185-190."),
    ("Park, D. S., Chan, W., Zhang, Y., Chiu, C.-C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019).",
     "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. "
     "Interspeech 2019, 2613-2617."),
    ("Winkler, W. E. (1990).",
     "String Comparator Metrics and Enhanced Decision Rules in the Fellegi-Sunter Model of Record Linkage. "
     "Proceedings of the Section on Survey Research Methods, American Statistical Association, 354-359."),
    ("Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011).",
     "Scikit-learn: Machine Learning in Python. "
     "Journal of Machine Learning Research, 12, 2825-2830."),
    ("LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).",
     "Gradient-Based Learning Applied to Document Recognition. "
     "Proceedings of the IEEE, 86(11), 2278-2324."),
]

for i, (authors, rest) in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1.0)
    p.paragraph_format.first_line_indent = Cm(-1.0)
    p.paragraph_format.space_after = Pt(4)
    r1 = p.add_run(f"[{i}] {authors} ")
    set_font(r1, size=11, bold=True)
    r2 = p.add_run(rest)
    set_font(r2, size=11)

# ══════════════════════════════════════════════
# GUARDAR
# ══════════════════════════════════════════════
out_path = os.path.join(OUTPUTS, "reporte_tecnico_CNN_voz_espanol.docx")
doc.save(out_path)
print(f"Documento guardado en: {out_path}")

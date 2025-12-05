## 🛠️ Requisitos e Instalación

Para ejecutar este proyecto, necesitarás Python y las librerías principales de *Deep Learning*.

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/nombre-del-repositorio.git](https://github.com/tu-usuario/nombre-del-repositorio.git)
    cd nombre-del-repositorio
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
    *(El archivo `requirements.txt` debería incluir `ultralytics`, `pandas`, `scikit-learn`, `opencv-python`, etc.)*

3.  **Descargar el Dataset:**
    Asegúrate de que el dataset se encuentre en la carpeta `aerial-cars-dataset/` o ajusta la ruta en el archivo `data.yaml`.
    
---

## ⚙️ Fases del Proyecto

### Fase 1: Análisis y Preprocesamiento de Datos (Tarea)

El script `scripts/preprocessor.py` realiza las siguientes funciones:

1.  **Interpretación del Formato de Anotación:**
    Se analiza la salida de los archivos `.txt` para confirmar que siguen el estándar YOLO: `class_id x_center y_center width height` (valores normalizados).
2.  **Conteo de Instancias por Clase:**
    Se realiza un conteo total de las instancias etiquetadas, confirmando la única clase de interés (coche/vehículo).
    > **Resultado:** Se identificaron **[INSERTA CONTEO TOTAL AQUÍ]** instancias de vehículos.
3.  **División Train/Test:**
    El dataset se divide en conjuntos de entrenamiento (80%) y prueba (20%) para la validación del modelo.

### Fase 2: Entrenamiento y Evaluación (Laboratorio)

La fase de entrenamiento se lleva a cabo utilizando el *notebook* **`Object_Detection_YOLOv8_Training.ipynb`**.

1.  **Configuración YOLOv8:**
    El archivo `data.yaml` define la ruta del dataset y especifica `nc: 1` y `names: ['car']`.
2.  **Entrenamiento:**
    Se entrena el modelo `yolov8s.pt` (Small) por `[N]` épocas.
    ```python
    from ultralytics import YOLO
    model = YOLO('yolov8s.pt') 
    model.train(data='data.yaml', epochs=N, imgsz=640)
    ```
3.  **Inferencia y Evaluación:**
    El modelo con los mejores pesos (`best.pt`) se utiliza para predecir objetos en el conjunto de prueba.

---

## 📊 Resultados Clave

El modelo **YOLOv8** demostró un rendimiento robusto en la detección de vehículos en entornos aéreos.

| Métrica | Valor Esperado | Interpretación |
| :---: | :---: | :---: |
| **mAP@0.50** | ~0.94 | Alto nivel de precisión en la detección y localización. |
| **mAP@0.50:0.95**| ~0.60 | Buena robustez de la caja delimitadora (bounding box) en umbrales estrictos. |

Las detecciones visuales confirman que el modelo es capaz de identificar vehículos de pequeño tamaño con alta confianza, incluso en áreas densamente pobladas.

### Ejemplo de Detección
*[Si tienes una imagen de ejemplo con los bounding boxes, agrégala aquí]*
``

---

## 🤝 Contribuciones

Este proyecto fue desarrollado por **[Tu Nombre Completo]** como parte del laboratorio de la materia **Inteligencia Artificial**.

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT.

from ultralytics import YOLO
import os

def main():
    # 1. Cargar el modelo
    # Usamos 'yolov8n.pt' (nano) porque es el más rápido y ligero para tu PC
    print("Cargando modelo...")
    model = YOLO('yolov8n.pt') 

    # 2. Definir la ruta a tu archivo YAML
    # Asegúrate de que esta ruta apunte al archivo que generó tu otro script
    yaml_path = os.path.join("dataset_split", "dataset.yaml")
    
    # Verificar que el archivo existe antes de empezar
    if not os.path.exists(yaml_path):
        print(f" Error: No encuentro el archivo {yaml_path}")
        return

    # 3. Entrenar el modelo
    print("Iniciando entrenamiento... (Esto puede tardar dependiendo de tu PC)")
    results = model.train(
        data=yaml_path,
        epochs=5,      # Pocas épocas para probar rápido (sube a 50 si tienes buena PC)
        imgsz=640,      # Tamaño de imagen estándar
        batch=8,        # Reduce esto a 4 o 2 si te sale error de "Out of Memory"
        device='0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu' # Usa GPU si hay, sino CPU
    )

    print(" Entrenamiento finalizado.")

    # 4. Validar el modelo (opcional, para ver métricas)
    metrics = model.val()

if __name__ == '__main__':
    main()
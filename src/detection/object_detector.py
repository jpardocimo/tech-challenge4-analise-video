"""
Módulo para detecção de objetos usando YOLO
"""

from typing import List, Dict, Tuple
from config.settings import INTEREST_OBJECT_CLASSES, OBJECT_DETECTION_CONFIDENCE


def detect_objects(frame, yolo_model) -> List[Dict]:
    """
    Detecta objetos de interesse no frame

    Objetos detectados:
    - Celular (cell phone)
    - Livro (book)
    - Laptop
    - Xícara (cup)
    - Garrafa (bottle)
    - Mesa (dining table)
    - Cadeira (chair)

    Args:
        frame: Frame de vídeo (numpy array)
        yolo_model: Modelo YOLO carregado

    Returns:
        Lista de dicionários com formato:
        [
            {'label': 'cell phone', 'bbox': (x1, y1, x2, y2)},
            ...
        ]
    """
    results = yolo_model.predict(
        frame,
        classes=INTEREST_OBJECT_CLASSES,
        conf=OBJECT_DETECTION_CONFIDENCE,
        verbose=False
    )

    objects = []

    if results:
        names = results[0].names
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            label = names[class_id]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            objects.append({
                'label': label,
                'bbox': (x1, y1, x2, y2)
            })

    return objects
